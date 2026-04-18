#!/bin/bash

# Create necessary directories
mkdir -p logs
mkdir -p models

echo "------------------------------------------------"
echo "Starting Training on Synthesized Data..."
echo "------------------------------------------------"

# -u flag allows real-time logging to the text file
python -u src/brain_mri_qc/train_synthesized.py > logs/training_log_synthetic.txt 2>&1

echo "Synthetic training complete. Weights saved in models/."