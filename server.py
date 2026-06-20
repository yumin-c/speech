"""
Flask server for Alzheimer speech demo.

Usage:
  python server.py                   # http://localhost:5000
  python server.py --port 7860
  python server.py --fusion          # use Late Fusion model
"""

import argparse
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import librosa
from flask import Flask, request, jsonify, send_from_directory
from transformers import Wav2Vec2Processor

sys.path.insert(0, str(Path(__file__).parent))
from features import extract, TARGET_SR
from model import AlzheimerFusionModel, WAV2VEC_MODEL

CKPT_DIR      = Path(__file__).parent / "checkpoints"
WEB_DIR       = Path(__file__).parent / "web"
MAX_AUDIO_LEN = TARGET_SR * 10

app = Flask(__name__, static_folder=str(WEB_DIR))

# ── globals set at startup ────────────────────────────────────────────────────
DEVICE        = None
PROCESSOR     = None
W2V_MODEL     = None
FUSION_MODEL  = None
ACOU_CLF      = None   # standalone acoustic classifier for late fusion
USE_FUSION    = False


def load_models(use_fusion: bool = False):
    global DEVICE, PROCESSOR, W2V_MODEL, FUSION_MODEL, ACOU_CLF, USE_FUSION
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    PROCESSOR = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL)

    ckpt = torch.load(CKPT_DIR / "best_lora.pt", map_location=DEVICE, weights_only=False)
    W2V_MODEL = AlzheimerFusionModel(strategy="lora").to(DEVICE)
    W2V_MODEL.load_state_dict(ckpt["model_state"])
    W2V_MODEL.eval()
    print(f"Loaded best_lora.pt  (epoch={ckpt.get('epoch','?')}, val_f1={ckpt.get('val_f1',0):.3f})")

    USE_FUSION = use_fusion
    if use_fusion:
        fp = CKPT_DIR / "best_fusion.pt"
        if fp.exists():
            from train_fusion import LateFusionMLP
            fc = torch.load(fp, map_location=DEVICE, weights_only=False)
            FUSION_MODEL = LateFusionMLP(in_dim=6, hidden=64).to(DEVICE)
            FUSION_MODEL.load_state_dict(fc["model_state"])
            FUSION_MODEL.eval()
            print(f"Loaded best_fusion.pt  (val_f1={fc.get('val_f1',0):.3f})")
        else:
            print("best_fusion.pt not found — falling back to Wav2Vec2 only")
            USE_FUSION = False

        # Load standalone acoustic classifier for real-time acoustic probs
        ap = CKPT_DIR / "best_acoustic.pt"
        if ap.exists():
            from train_fusion import AcousticClassifier
            ac = torch.load(ap, map_location=DEVICE, weights_only=False)
            ACOU_CLF = AcousticClassifier().to(DEVICE)
            ACOU_CLF.load_state_dict(ac["model_state"])
            ACOU_CLF.eval()
            print(f"Loaded best_acoustic.pt  (val_f1={ac.get('val_f1',0):.3f}, val_auc={ac.get('val_auc',0):.3f})")
        else:
            print("best_acoustic.pt not found — acoustic branch will use 0.5 fallback")


# ── routes ────────────────────────────────────────────────────────────────────

FIGURES_DIR = Path(__file__).parent / "figures"


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/figures/<path:filename>")
def figures(filename):
    return send_from_directory(FIGURES_DIR, filename)


@app.route("/stats")
def stats():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True
        ).strip().split("\n")
        gpus = []
        for line in out:
            name, used, total, util = [x.strip() for x in line.split(",")]
            used_gb  = round(int(used)  / 1024, 1)
            total_gb = round(int(total) / 1024, 1)
            pct      = round(int(used) / int(total) * 100, 1)
            gpus.append({
                "name": name, "used_gb": used_gb,
                "total_gb": total_gb, "pct": pct,
                "util_pct": int(util),
            })
        return jsonify({"available": True, "gpus": gpus})
    except Exception as e:
        return jsonify({"available": False, "error": str(e)})


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "no audio file"}), 400

    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        y, _ = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        return jsonify({"error": f"audio load failed: {e}"}), 400
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if len(y) < MAX_AUDIO_LEN:
        y = np.pad(y, (0, MAX_AUDIO_LEN - len(y)))
    else:
        y = y[:MAX_AUDIO_LEN]

    acoustic = extract(y, TARGET_SR)

    with torch.inference_mode():
        inp = PROCESSOR(y, sampling_rate=TARGET_SR, return_tensors="pt", padding=False)
        iv  = inp.input_values.to(DEVICE)
        am  = torch.ones_like(iv, dtype=torch.long)
        af  = torch.tensor(acoustic, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits = W2V_MODEL(iv, am, af)
        w2v_probs = torch.softmax(logits, -1).squeeze(0).cpu().numpy()

    if USE_FUSION and FUSION_MODEL is not None:
        # Acoustic classifier branch (real-time)
        if ACOU_CLF is not None:
            with torch.inference_mode():
                ac_logits = ACOU_CLF(af)
                ac_probs  = torch.softmax(ac_logits, -1).squeeze(0).cpu().numpy()
        else:
            ac_probs = np.array([0.5, 0.5], dtype=np.float32)

        # Gemma4 not available at inference time — use neutral prior
        g4_probs = np.array([0.5, 0.5], dtype=np.float32)

        x = torch.tensor(
            np.concatenate([w2v_probs, g4_probs, ac_probs]).astype(np.float32),
            device=DEVICE
        ).unsqueeze(0)
        with torch.inference_mode():
            probs = torch.softmax(FUSION_MODEL(x), -1).squeeze(0).cpu().numpy()
        model_tag = "Late Fusion (w2v + acoustic)"
    else:
        probs     = w2v_probs
        model_tag = "Wav2Vec2 + LoRA"

    p_healthy   = float(probs[0])
    p_alzheimer = float(probs[1])
    label       = "Alzheimer" if p_alzheimer > 0.5 else "Healthy"

    return jsonify({
        "label":       label,
        "p_healthy":   round(p_healthy, 4),
        "p_alzheimer": round(p_alzheimer, 4),
        "confidence":  round(max(p_healthy, p_alzheimer), 4),
        "model":       model_tag,
    })


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",   type=int,  default=5000)
    parser.add_argument("--host",   type=str,  default="0.0.0.0")
    parser.add_argument("--fusion", action="store_true")
    parser.add_argument("--https",  action="store_true",
                        help="Enable HTTPS with self-signed cert (required for microphone on non-localhost)")
    args = parser.parse_args()

    load_models(use_fusion=args.fusion)

    ssl_context = None
    if args.https:
        import ssl, os
        cert_dir = Path(__file__).parent / ".ssl"
        cert_dir.mkdir(exist_ok=True)
        cert_file = cert_dir / "cert.pem"
        key_file  = cert_dir / "key.pem"
        if not cert_file.exists():
            os.system(
                f'openssl req -x509 -newkey rsa:2048 -keyout {key_file} '
                f'-out {cert_file} -days 365 -nodes '
                f'-subj "/CN=localhost" 2>/dev/null'
            )
        ssl_context = (str(cert_file), str(key_file))
        print(f"HTTPS enabled — access via https://<IP>:{args.port}")
        print("(Browser will warn about self-signed cert — click 'Advanced > Proceed')")
    else:
        print(f"HTTP — microphone works on localhost only")
        print(f"Use --https flag to enable microphone from other devices")

    app.run(host=args.host, port=args.port, debug=False, ssl_context=ssl_context)
