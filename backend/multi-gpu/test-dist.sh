#!/bin/bash

# --- Configuration ---
# Set the memory threshold for a GPU to be considered "idle".
# A value of 4096 MiB is usually a safe bet to exclude GPUs running desktops or small processes.
MEM_THRESHOLD_MB=4096

# --- Script Logic ---
set -e

echo "Searching for idle GPUs with memory usage < ${MEM_THRESHOLD_MB}MiB for testing..."

# Use nvidia-smi to query GPU index and memory usage.
IDLE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v threshold="$MEM_THRESHOLD_MB" '$2 < threshold {print $1}')

# Check if any idle GPUs were found.
if [ -z "$IDLE_GPUS" ]; then
  echo "Error: No idle GPUs found for testing. Exiting."
  exit 1
fi

# Convert the newline-separated list of IDs into a comma-separated string
CUDA_VISIBLE_DEVICES=$(echo "$IDLE_GPUS" | tr '\n' ',' | sed 's/,$//')
# Count how many idle GPUs were found.
NUM_GPUS=$(echo "$IDLE_GPUS" | wc -l)

echo "Found ${NUM_GPUS} idle GPUs: [${CUDA_VISIBLE_DEVICES}]"
echo "Starting distributed testing..."
echo "---"

# Set environment variables and launch the testing script.
# `$@` passes all command-line arguments from this script to the python script.
TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
torchrun --nproc_per_node=${NUM_GPUS} dist_test.py \
    --model_path ../../post-trained-model/bert-snli-no-adversarial/ \
    --dataset_name snli \
    --batch_size 512 \
    --epsilon 1.0 \
    --results_file ../../post-trained-model/bert-snli-no-adversarial/test_results.json
