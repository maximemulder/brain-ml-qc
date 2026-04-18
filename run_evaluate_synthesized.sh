#!/bin/bash

# Create necessary directories
mkdir -p logs
mkdir -p models

echo "------------------------------------------------"
echo "Starting Evaluation on Synthesized Data..."
echo "------------------------------------------------"

# -u flag allows real-time logging to the text file
python -u src/brain_mri_qc/evaluate_synthesized.py

echo "Synthetic evaluation complete."