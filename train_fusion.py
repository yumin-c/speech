"""
Late fusion ablation training.

Trains one MLP per modality combination:
  w2v / gemma4 / acoustic / w2v+gemma4 / w2v+acoustic / gemma4+acoustic / full

Usage:
  python train_fusion.py                     # all 7 ablations
  python train_fusion.py --combo full        # single combo
  python train_fusion.py --epochs 100 --lr 1e-3
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, roc_auc_score
import librosa
from transformers import Wav2Vec2Processor

from features import extract, TARGET_SR, FEATURE_DIM
from model import AlzheimerFusionModel, WAV2VEC_MODEL

DATA_DIR = Path("/home/ymcheong/extdata3/YMC/gemma_speech/data")
CKPT_DIR = Path("/home/ymcheong/extdata3/YMC/gemma_speech/checkpoints")
LOG_PATH = Path("/home/ymcheong/extdata3/YMC/gemma_speech/fusion_ablation_log.json")
MAX_AUDIO_LEN = TARGET_SR * 10

# modality slice indices within 6-dim vector [w2v(0:2), gemma4(2:4), acou(4:6)]
COMBOS = {
    "w2v":           [0, 1],
    "gemma4":        [2, 3],
    "acoustic":      [4, 5],
    "w2v+gemma4":    [0, 1, 2, 3],
    "w2v+acoustic":  [0, 1, 4, 5],
    "gemma4+acoustic": [2, 3, 4, 5],
    "full":          [0, 1, 2, 3, 4, 5],
}


# ── Models ────────────────────────────────────────────────────────────────────

class LateFusionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class AcousticClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


# ── Feature pre-computation ───────────────────────────────────────────────────

def build_feature_matrix(split, w2v_model, w2v_processor, acou_clf, device):
    """Returns (N, 6) tensor [w2v_p, g4_p, ac_p] and (N,) labels for a split."""
    samples = []
    for label, cls in [(1, "alzheimer"), (0, "healthy")]:
        d = DATA_DIR / split / cls
        if d.exists():
            for p in sorted(d.glob("*.wav")):
                samples.append((str(p), label))

    g4_cache = DATA_DIR / split / "gemma4_probs.npz"
    if not g4_cache.exists():
        raise FileNotFoundError(f"Gemma4 cache missing: {g4_cache}\nRun cache_gemma4_logits.py first.")
    cache = np.load(g4_cache, allow_pickle=True)
    gemma4_map = {p: pr for p, pr in zip(cache["paths"], cache["probs"])}

    rows, labels = [], []
    w2v_model.eval(); acou_clf.eval()
    with torch.no_grad():
        for path, label in samples:
            y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
            y = y[:MAX_AUDIO_LEN] if len(y) >= MAX_AUDIO_LEN else np.pad(y, (0, MAX_AUDIO_LEN - len(y)))

            inp = w2v_processor(y, sampling_rate=TARGET_SR, return_tensors="pt", padding=False)
            iv = inp.input_values.to(device)
            am = torch.ones_like(iv, dtype=torch.long)
            af = torch.tensor(extract(y), dtype=torch.float32, device=device).unsqueeze(0)

            w2v_p = torch.softmax(w2v_model(iv, am, af), -1).squeeze(0).cpu().numpy()
            ac_p  = torch.softmax(acou_clf(af), -1).squeeze(0).cpu().numpy()
            g4_p  = gemma4_map.get(path, np.array([0.5, 0.5], dtype=np.float32))

            rows.append(np.concatenate([w2v_p, g4_p, ac_p]).astype(np.float32))
            labels.append(label)

    X = torch.tensor(np.array(rows), dtype=torch.float32, device=device)
    Y = torch.tensor(labels, dtype=torch.long, device=device)
    print(f"  {split}: {len(labels)} samples  (alz={sum(labels)}, hea={len(labels)-sum(labels)})")
    return X, Y


# ── Eval ──────────────────────────────────────────────────────────────────────

def evaluate(model, X, Y, indices, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(X[:, indices])
        loss = criterion(logits, Y).item()
        probs = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        preds = logits.argmax(-1).cpu().numpy()
        labels = Y.cpu().numpy()
    acc = float(np.mean(preds == labels))
    f1  = f1_score(labels, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.0
    return loss, acc, f1, auc, probs, preds


# ── Single-combo training ─────────────────────────────────────────────────────

def train_combo(combo_name, indices, tr_X, tr_Y, va_X, va_Y, args, device):
    in_dim = len(indices)
    idx = torch.tensor(indices, device=device)

    n_alz = (tr_Y == 1).sum().item()
    n_hea = (tr_Y == 0).sum().item()
    w = torch.tensor([n_alz / (n_alz + n_hea), n_hea / (n_alz + n_hea)],
                     dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)

    model = LateFusionMLP(in_dim=in_dim, hidden=64).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sch = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    tr_ds = TensorDataset(tr_X[:, idx], tr_Y)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)

    best_f1, best_state, best_metrics = 0.0, None, {}
    log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0
        for xb, yb in tr_ld:
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()
            total_loss += criterion(model(xb), yb).item()
        sch.step()

        tr_loss = total_loss / len(tr_ld)
        va_loss, va_acc, va_f1, va_auc, _, _ = evaluate(model, va_X, va_Y, idx, criterion)

        entry = dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss,
                     val_acc=va_acc, val_f1=va_f1, val_auc=va_auc)
        log.append(entry)

        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = dict(val_acc=va_acc, val_f1=va_f1, val_auc=va_auc, epoch=epoch)

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"  [{combo_name}] ep{epoch:3d} | tr={tr_loss:.4f} va={va_loss:.4f} "
                  f"acc={va_acc:.3f} f1={va_f1:.3f} auc={va_auc:.3f} | {time.time()-t0:.1f}s")

    # save best checkpoint
    model.load_state_dict(best_state)
    ckpt_path = CKPT_DIR / f"fusion_{combo_name.replace('+','_')}.pt"
    torch.save({"epoch": best_metrics["epoch"], "model_state": best_state,
                "val_f1": best_metrics["val_f1"], "val_auc": best_metrics["val_auc"],
                "combo": combo_name, "indices": indices}, ckpt_path)
    print(f"  [{combo_name}] Best F1={best_f1:.3f}  saved → {ckpt_path.name}")

    # keep best_fusion.pt as the full-fusion checkpoint for server compatibility
    if combo_name == "full":
        import shutil
        shutil.copy(ckpt_path, CKPT_DIR / "best_fusion.pt")

    return best_metrics, log


# ── Main ──────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Load frozen Wav2Vec2
    print("Loading Wav2Vec2 (best_lora.pt)...")
    w2v_processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL)
    w2v_model = AlzheimerFusionModel(strategy="lora").to(device)
    w2v_model.load_state_dict(
        torch.load(CKPT_DIR / "best_lora.pt", map_location=device, weights_only=False)["model_state"])
    for p in w2v_model.parameters():
        p.requires_grad = False

    # Train acoustic classifier
    print("\nTraining acoustic classifier...")
    acou_clf = AcousticClassifier().to(device)
    acou_opt = AdamW(acou_clf.parameters(), lr=1e-3)
    acou_crit = nn.CrossEntropyLoss()

    def _load_acoustic(split):
        feats, lbls = [], []
        for label, cls in [(1, "alzheimer"), (0, "healthy")]:
            d = DATA_DIR / split / cls
            if not d.exists(): continue
            for p in sorted(d.glob("*.wav")):
                y, _ = librosa.load(str(p), sr=TARGET_SR, mono=True)
                y = y[:MAX_AUDIO_LEN] if len(y) >= MAX_AUDIO_LEN else np.pad(y, (0, MAX_AUDIO_LEN - len(y)))
                feats.append(extract(y)); lbls.append(label)
        return (torch.tensor(np.array(feats), dtype=torch.float32, device=device),
                torch.tensor(lbls, dtype=torch.long, device=device))

    tr_acX, tr_acY = _load_acoustic("train")
    print(f"  Acoustic train: {len(tr_acY)} samples")
    ac_ds = TensorDataset(tr_acX, tr_acY)
    ac_ld = DataLoader(ac_ds, batch_size=256, shuffle=True)
    for ep in range(30):
        acou_clf.train()
        for af, lbl in ac_ld:
            acou_opt.zero_grad()
            acou_crit(acou_clf(af), lbl).backward()
            acou_opt.step()
    for p in acou_clf.parameters():
        p.requires_grad = False

    # Evaluate acoustic classifier on val
    va_acX, va_acY = _load_acoustic("val")
    with torch.no_grad():
        ac_logits = acou_clf(va_acX)
        ac_preds = ac_logits.argmax(-1).cpu().numpy()
        ac_labels = va_acY.cpu().numpy()
        ac_probs_val = torch.softmax(ac_logits, -1)[:, 1].cpu().numpy()
    ac_f1  = f1_score(ac_labels, ac_preds, average="binary", zero_division=0)
    ac_auc = roc_auc_score(ac_labels, ac_probs_val)
    print(f"  Acoustic classifier val: F1={ac_f1:.3f}  AUC={ac_auc:.3f}")
    torch.save({"model_state": acou_clf.state_dict(), "val_f1": ac_f1, "val_auc": ac_auc},
               CKPT_DIR / "best_acoustic.pt")

    # Pre-compute full 6-dim feature matrices
    print("\nPre-computing fusion feature matrices...")
    tr_X, tr_Y = build_feature_matrix("train", w2v_model, w2v_processor, acou_clf, device)
    va_X, va_Y = build_feature_matrix("val",   w2v_model, w2v_processor, acou_clf, device)

    # Ablation loop
    combos_to_run = ([args.combo] if args.combo != "all"
                     else list(COMBOS.keys()))
    print(f"\nRunning ablations: {combos_to_run}")

    all_results = {}
    all_logs = {}

    # W2V-only val metrics come directly from the frozen model (no MLP needed)
    # but we still train a 2-dim MLP wrapper for fair comparison
    for combo in combos_to_run:
        print(f"\n── {combo} (in_dim={len(COMBOS[combo])}) ──")
        metrics, log = train_combo(
            combo, COMBOS[combo], tr_X, tr_Y, va_X, va_Y, args, device)
        all_results[combo] = metrics
        all_logs[combo] = log

    # Save summary
    summary = {"results": all_results, "args": vars(args)}
    LOG_PATH.write_text(json.dumps(summary, indent=2))

    print("\n\n═══ Ablation Summary (Val) ═══")
    print(f"{'Combo':<20} {'Acc':>6} {'F1':>6} {'AUC':>6}  {'Epoch':>5}")
    print("─" * 50)
    for combo, m in all_results.items():
        print(f"{combo:<20} {m['val_acc']:>6.3f} {m['val_f1']:>6.3f} {m['val_auc']:>6.3f}  {m['epoch']:>5}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--combo",      type=str,   default="all",
                        choices=["all"] + list(COMBOS.keys()))
    args = parser.parse_args()
    train(args)
