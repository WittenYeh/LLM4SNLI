# dist_train.py

import torch
import argparse
import os
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from data_loader import get_nli_dataloader
from fgm_attack import FGM
from evaluator import evaluate, evaluate_adversarial

def setup_ddp():
    """
    Initializes the distributed process group.
    `torchrun` automatically sets the necessary environment variables (`MASTER_ADDR`, `MASTER_PORT`, etc.),
    so `init_process_group` can be called without arguments.
    `backend="nccl"` is the standard, highly-optimized backend for NVIDIA GPUs.
    """
    dist.init_process_group(backend="nccl")

def cleanup_ddp():
    """Cleans up the distributed process group, releasing resources."""
    dist.destroy_process_group()

# --- Main Training Function ---
def main():
    # --- DDP Setup ---
    # This must be the first step in the main function to set up communication between processes.
    setup_ddp()
    # Fetch process-specific information from environment variables set by `torchrun`.
    # RANK: The global, unique identifier for the current process across all nodes.
    # LOCAL_RANK: The local identifier for the process on the current node (maps directly to GPU ID).
    # WORLD_SIZE: The total number of processes (GPUs) participating in the training.
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # --- Argument Parsing ---
    # This section defines the command-line arguments that configure the training run.
    parser = argparse.ArgumentParser(description="Distributed training of a text classification model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the folder containing the pretrained model and tokenizer.")
    parser.add_argument("--dataset_name", type=str, default="snli", choices=["snli", "mnli"], help="Name of the NLI dataset to use.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size PER GPU for training and evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before performing an optimization step.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--do_adversarial_training", action='store_true', help="If set, perform FGM adversarial training.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Perturbation size (epsilon) for FGM attack.")
    parser.add_argument("--output_dir", type=str, default="./trained_model", help="Directory where the final model will be saved.")
    args = parser.parse_args()

    # --- Device Setup ---
    # Each process is pinned to a specific GPU using its `local_rank`.
    device = torch.device(f"cuda:{local_rank}")
    # Only the main process (rank 0) should print general information to avoid cluttered logs.
    if rank == 0:
        print(f"Starting DDP training on {world_size} GPUs.")
        # Calculate the true, effective batch size across all GPUs and accumulation steps.
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
        print(f"Per-GPU Batch Size: {args.batch_size}")
        print(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
        print(f"Effective Total Batch Size: {effective_batch_size}")

    # --- Load Model and Tokenizer ---
    # The number of labels is fixed for NLI datasets.
    num_labels = 3
    # Each process loads the tokenizer and model independently.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels).to(device)
    
    # Wrap the model with DDP. This is the core of distributed training.
    # DDP handles gradient synchronization (all-reduce) across all processes after the backward pass.
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=args.do_adversarial_training)
    # `find_unused_parameters=True` is sometimes necessary if parts of the model (e.g., in conditional logic)
    # do not receive gradients in a given forward pass. FGM might cause this, so we set it defensively.

    # --- Load and Distribute Data ---
    if rank == 0:
        print("Loading and preparing dataset...")
    # The data_loader is called with `return_datasets=True` to get raw Hugging Face Dataset objects.
    train_dataset, eval_dataset, test_dataset = get_nli_dataloader(
        args.dataset_name, tokenizer, args.batch_size, return_datasets=True
    )
    # The DistributedSampler ensures that each GPU receives a different, non-overlapping subset of the data.
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # The DataLoader uses the sampler. `shuffle` must be False for the DataLoader itself,
    # as shuffling is now handled by the DistributedSampler.
    # `num_workers` > 0 enables multi-process data loading to prevent the GPU from waiting for data.
    # `pin_memory=True` speeds up CPU-to-GPU data transfer.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, sampler=eval_sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)
    
    # --- Optimizer, Scheduler, and Scaler ---
    # AdamW is the standard optimizer for Transformer models.
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # The number of actual optimization steps is reduced by the number of accumulation steps.
    num_training_steps = args.epochs * len(train_loader) // args.gradient_accumulation_steps
    # A linear learning rate decay scheduler is a common and effective choice.
    lr_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)
    # GradScaler helps manage the scaling of gradients for mixed precision.
    scaler = GradScaler()

    # --- Adversarial Training Setup ---
    if args.do_adversarial_training:
        # FGM now operates on the DDP-wrapped model.
        fgm = FGM(model)
        if rank == 0:
            print("Adversarial training (FGM) is ENABLED.")
    
    # --- Training Loop ---
    for epoch in range(args.epochs):
        # This is crucial for ensuring the data is shuffled differently in each epoch across all GPUs.
        train_sampler.set_epoch(epoch)
        # Set the model to training mode (enables dropout, etc.).
        model.train()
        
        # The progress bar is only created and displayed on the main process to avoid messy output.
        data_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(rank != 0))
        
        for i, batch in enumerate(data_iterator):
            # Move data to the assigned GPU. `non_blocking=True` can slightly speed up transfer by overlapping it with computation.
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device, non_blocking=True)
            
            # The `autocast` context manager enables Automatic Mixed Precision for this block.
            # It automatically casts operations to FP16 where safe, reducing memory and speeding up computation on Tensor Cores.
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                # When using gradient accumulation, the loss for each mini-batch must be averaged.
                loss = loss / args.gradient_accumulation_steps
            
            # `scaler.scale(loss)` multiplies the loss by a scaling factor to prevent FP16 gradients from vanishing.
            # `backward()` then computes the scaled gradients.
            scaler.scale(loss).backward()

            if args.do_adversarial_training:
                # The adversarial step follows the same pattern of autocasting and scaled backpropagation.
                fgm.attack(epsilon=args.epsilon)
                with autocast(device_type='cuda', dtype=torch.float16):
                    adv_outputs = model(**inputs, labels=labels)
                    adv_loss = adv_outputs.loss
                    adv_loss = adv_loss / args.gradient_accumulation_steps
                scaler.scale(adv_loss).backward()
                fgm.restore()

            # The optimizer step is only performed after enough gradients have been accumulated.
            if (i + 1) % args.gradient_accumulation_steps == 0:
                # `scaler.step` first unscales the gradients. If no inf/NaNs are found, it calls `optimizer.step()`.
                scaler.step(optimizer)
                # `scaler.update` updates the scaling factor for the next iteration.
                scaler.update()
                # The learning rate scheduler is also stepped only when the optimizer is stepped.
                lr_scheduler.step()
                # Gradients must be cleared before the next accumulation cycle.
                optimizer.zero_grad()

            if rank == 0:
                # Display the non-normalized loss for better readability.
                data_iterator.set_postfix(loss=(loss.item() * args.gradient_accumulation_steps))
        
        # --- Evaluation after each epoch ---
        # The DDP-aware `evaluate` function handles gathering results from all GPUs.
        eval_results = evaluate(model, eval_loader, device)
        # `dist.barrier()` acts as a synchronization point, ensuring all processes have finished
        # their work (in this case, evaluation) before the main process proceeds to print results.
        dist.barrier()
        
        if rank == 0:
            print(f"\n--- Evaluating after Epoch {epoch+1} ---")
            # `.get('key', 0)` is a safe way to access the dictionary, avoiding errors if a key is missing.
            print(f"Validation Results: Accuracy={eval_results.get('accuracy', 0):.4f}, F1={eval_results.get('f1', 0):.4f}")

    # --- Final Evaluation and Saving (only on main process) ---
    if rank == 0:
        print("\n--- Final Evaluation on Test Set ---")
        test_results_clean = evaluate(model, test_loader, device)
        print(f"Clean Test Results: Accuracy={test_results_clean.get('accuracy', 0):.4f}, F1={test_results_clean.get('f1', 0):.4f}")

        if args.do_adversarial_training:
            print("\n--- Evaluating on ADVERSARIAL test data ---")
            test_results_adv = evaluate_adversarial(model, test_loader, device, fgm, args.epsilon)
            print(f"Adversarial Test Results: Robust Accuracy={test_results_adv.get('robust_accuracy', 0):.4f}, Adversarial Success Rate={test_results_adv.get('adversarial_success_rate', 0):.4f}")
        
        if args.output_dir:
            print(f"\n--- Saving final model to {args.output_dir} ---")
            os.makedirs(args.output_dir, exist_ok=True)
            # When saving a DDP-wrapped model, it's crucial to save the underlying `model.module`.
            # This saves the actual model weights, not the DDP wrapper, ensuring it can be loaded later as a standard model.
            model.module.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print("Model and tokenizer saved successfully.")
    
    # --- Cleanup ---
    # Properly shut down the distributed environment.
    cleanup_ddp()

# --- Script Entry Point ---
if __name__ == "__main__":
    main()