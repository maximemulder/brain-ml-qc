#!/bin/bash

# Create necessary directories
mkdir -p logs
mkdir -p models

echo "------------------------------------------------"
echo "Starting Evaluation on ABIDE I (Confidence-Weighted)..."
echo "------------------------------------------------"

# -u flag allows real-time logging to the text file
python -u src/brain_mri_qc/evaluate_abide_confidence.py

echo "ABIDE I (Confidence-Weighted) evaluation complete."
