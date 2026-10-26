#!/bin/bash
# Quick test to verify model saving works

cd "$(dirname "$0")/src"
source /home/vishwas/Desktop/RDS/.venv/bin/activate

# Run a very short training (5 epochs) to test saving
python main.py -m BM3 -d baby 2>&1 | tee test_saving.log

echo ""
echo "========================================"
echo "Checking if model was saved..."
echo "========================================"
ls -lh saved/*.pth 2>/dev/null || echo "No models found in saved/"
