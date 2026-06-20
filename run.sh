#!/bin/bash
# Full pipeline: data prep → train → evaluate
# Usage: bash run.sh [strategy]
#   strategy: lora (default) | full | frozen
set -e

PYTHON=/home/ymcheong/.conda/envs/ym_voice/bin/python
STRATEGY=${1:-lora}
DIR=/home/ymcheong/extdata3/YMC/gemma_speech

cd $DIR

echo "=== [1/3] Data preparation ==="
$PYTHON prepare_data.py

echo ""
echo "=== [2/3] Training (strategy=$STRATEGY) ==="
$PYTHON train.py --strategy $STRATEGY --epochs 30 --batch_size 8 --lr 3e-4

echo ""
echo "=== [3/3] Evaluation ==="
$PYTHON evaluate.py --strategy $STRATEGY

echo ""
echo "Done. Figures in $DIR/figures/"
