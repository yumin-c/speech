# Alzheimer's disease speech classification

Binary Alzheimer classification from speech, using a Wav2Vec2 encoder fused with hand crafted acoustic features.

## Architecture

- `features.py`: extracts a 130 dim acoustic feature vector (MFCC, pitch, energy, spectral stats).
- `model.py`: Wav2Vec2 large encoder (LoRA, full fine tune, or frozen) fused with an acoustic MLP, then a binary classifier.
- `train_fusion.py`: late fusion ablation training across modality combinations (w2v, gemma4, acoustic, and pairs/full).
- `server.py`: Flask demo server with a `/predict` endpoint and a static web UI.
- `web/index.html`: browser based demo UI for live microphone recording and result sharing.

## Setup

```bash
python -m pip install torch transformers librosa flask numpy scikit-learn
```

## Usage

Run the full pipeline (data prep, training, evaluation):

```bash
bash run.sh [lora|full|frozen]
```

Start the demo server:

```bash
python server.py --fusion --https
```

Then open the printed URL in a browser to record audio and get a prediction.

## Figures

Training/evaluation plots (ROC curves, KDEs, loss curves, ablation comparisons) are written to `figures/`.
