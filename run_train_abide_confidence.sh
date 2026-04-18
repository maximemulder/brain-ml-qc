#!/bin/bash

# Ensure directories exist
mkdir -p logs
mkdir -p models

echo "------------------------------------------------"
echo "Starting Fine-tuning on ABIDE I (Confidence-Weighted)..."
echo "------------------------------------------------"

# Assuming you want to capture the output in your existing log file
python -u src/brain_mri_qc/train_abide_confidence.py > logs/training_log_confidence.txt 2>&1

echo "Fine-tuning complete. Check logs/training_log_confidence.txt for metrics."