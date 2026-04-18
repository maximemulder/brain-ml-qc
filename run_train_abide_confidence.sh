#!/bin/bash

# Ensure directories exist
mkdir -p logs
mkdir -p models

echo "------------------------------------------------"
echo "Starting Training on ABIDE I (Confidence-Weighted)"
echo "------------------------------------------------"

# Assuming you want to capture the output in your existing log file
python -u src/brain_mri_qc/train_abide_confidence.py

echo "Complete Training on ABIDE I (Confidence-Weighted). Weights saved in models/."