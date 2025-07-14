#!/bin/bash

# --- Configuration ---
# Set the memory threshold for a GPU to be considered "idle".
# If a GPU's memory usage is below this value (in MiB), it will be selected.
# A value of 4096 MiB is usually a safe bet to exclude GPUs running desktops or small processes.
MEM_THRESHOLD_MB=4096

# --- Script Logic ---
# `set -e` ensures the script will exit immediately if any command fails.
set -e

echo "Searching for idle GPUs with memory usage < ${MEM_THRESHOLD_MB}MiB..."

# Use nvidia-smi to query GPU index and memory usage.
# --format=csv,noheader,nounits makes the output easy to parse.
# Example output line: "0, 15" (index 0, 15MiB used)
IDLE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v threshold="$MEM_THRESHOLD_MB" '$2 < threshold {print $1}')

# Check if any idle GPUs were found.
if [ -z "$IDLE_GPUS" ]; then
  echo "Error: No idle GPUs found. Exiting."
  exit 1
fi

# Convert the newline-separated list of IDs into a comma-separated string for CUDA_VISIBLE_DEVICES.
# For example, "0\n2\n5" becomes "0,2,5".
CUDA_VISIBLE_DEVICES=$(echo "$IDLE_GPUS" | tr '\n' ',' | sed 's/,$//')
# Count how many idle GPUs were found.
NUM_GPUS=$(echo "$IDLE_GPUS" | wc -l)

echo "Found ${NUM_GPUS} idle GPUs: [${CUDA_VISIBLE_DEVICES}]"
echo "Starting distributed training..."
echo "---"

# Set the environment variables and launch the training script.
# CUDA_VISIBLE_DEVICES tells PyTorch which physical GPUs to use. They will be re-indexed from 0.
# TOKENIZERS_PARALLELISM=false avoids a common warning/deadlock in forked processes.
# `$@` passes all command-line arguments from this script to the python script.
TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
torchrun --nproc_per_node=${NUM_GPUS} dist_train.py \
    --model_path ../../models/bert-base-uncased \
    --dataset_name mnli \
    --epochs 5 \
    --batch_size 1024 \
    --lr 2e-5 \
    --output_dir ../../post-trained-model/bert-mnli-adversarial/ \
    --do_adversarial_training
