#!/bin/bash
# Test the trained BM3 model

cd "$(dirname "$0")/src"
source /home/vishwas/Desktop/RDS/.venv/bin/activate

python load_and_test.py \
    -c ../models/trained/baby/baby/BM3-Oct-20-2026-12-11-13.pth \
    -d baby \
    "$@"
