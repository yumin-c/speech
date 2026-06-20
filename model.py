"""
Fusion model for Alzheimer binary classification from speech.

Architecture:
  1. Wav2Vec2-large encoder with LoRA injected directly into attention layers
     → mean-pool hidden states → 1024-dim speech representation
  2. Acoustic MLP: 130-dim hand-crafted features → 256-dim
  3. Fusion: concat(1024, 256) → 512 → 2 (binary classifier)

Strategy options:
  - "lora"   : LoRA on q/k/v/out projections (default, ~1.2% params trainable)
  - "full"   : Full fine-tuning of Wav2Vec2
  - "frozen" : Freeze Wav2Vec2, train only acoustic MLP + classifier (baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

from features import FEATURE_DIM

WAV2VEC_MODEL = "facebook/wav2vec2-large-960h"
WAV2VEC_DIM = 1024
ACOUSTIC_HIDDEN = 256
FUSION_HIDDEN = 512
NUM_LABELS = 2
TRAINABLE_STRATEGY = "lora"


class LoRALinear(nn.Module):
    """Drop-in LoRA wrapper for nn.Linear."""

    def __init__(self, linear: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.1):
        super().__init__()
        self.linear = linear
        self.r = r
        self.scale = alpha / r
        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        # Freeze base linear
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scale * self.lora_B(self.lora_A(self.dropout(x)))


def inject_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, dropout: float = 0.1):
    """Replace q/k/v/out projections in Wav2Vec2 attention layers with LoRALinear."""
    target_names = {"q_proj", "k_proj", "v_proj", "out_proj"}
    for name, module in list(model.named_modules()):
        if any(name.endswith(f".{t}") for t in target_names):
            parent_name, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            original = getattr(parent, attr)
            if isinstance(original, nn.Linear):
                setattr(parent, attr, LoRALinear(original, r=r, alpha=alpha, dropout=dropout))
    return model


class AcousticMLP(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = ACOUSTIC_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AlzheimerFusionModel(nn.Module):
    def __init__(self, strategy: str = TRAINABLE_STRATEGY):
        super().__init__()
        self.strategy = strategy

        self.wav2vec = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL)

        if strategy == "lora":
            # Freeze all base params first
            for p in self.wav2vec.parameters():
                p.requires_grad = False
            # Inject trainable LoRA adapters
            inject_lora(self.wav2vec, r=16, alpha=32.0, dropout=0.1)
        elif strategy == "frozen":
            for p in self.wav2vec.parameters():
                p.requires_grad = False
        # "full": everything trainable (default from_pretrained)

        self.acoustic_mlp = AcousticMLP()
        fusion_in = WAV2VEC_DIM + ACOUSTIC_HIDDEN
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, FUSION_HIDDEN),
            nn.LayerNorm(FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(FUSION_HIDDEN, NUM_LABELS),
        )

    def forward(
        self,
        input_values: torch.Tensor,        # (B, T) raw waveform
        attention_mask: torch.Tensor,      # (B, T)
        acoustic_features: torch.Tensor,  # (B, FEATURE_DIM)
    ) -> torch.Tensor:
        w2v_out = self.wav2vec(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden = w2v_out.last_hidden_state  # (B, T', D)
        speech_repr = hidden.mean(dim=1)    # (B, D)

        acou_repr = self.acoustic_mlp(acoustic_features)
        fused = torch.cat([speech_repr, acou_repr], dim=-1)
        return self.classifier(fused)

    def print_trainable(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


if __name__ == "__main__":
    import sys
    strategy = sys.argv[1] if len(sys.argv) > 1 else "lora"
    model = AlzheimerFusionModel(strategy=strategy)
    model.print_trainable()
    B, T = 2, 16000 * 5
    logits = model(torch.randn(B, T), torch.ones(B, T, dtype=torch.long), torch.randn(B, FEATURE_DIM))
    print(f"Output shape: {logits.shape}")
