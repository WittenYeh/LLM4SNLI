# dist_train.py (MODIFIED - with metrics logging and visualization)

import torch
import argparse
import os
import json # Import json for saving metrics
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from data_loader import get_nli_dataloader
from fgm_attack import FGM
from evaluator import evaluate
from visualizer import plot_and_save_metrics # Import the new visualizer function

def setup_ddp():
    dist.init_process_group(backend="nccl")

def cleanup_ddp():
    dist.destroy_process_group()

def main():
    setup_ddp()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Argument parsing remains the same
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

    device = torch.device(f"cuda:{local_rank}")
    if rank == 0:
        print(f"Starting DDP training on {world_size} GPUs.")
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
        print(f"Effective Total Batch Size: {effective_batch_size}")

    num_labels = 3
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if rank == 0:
        print("Loading and preparing dataset...")
    train_dataset, eval_dataset, _ = get_nli_dataloader(
        args.dataset_name, tokenizer, args.batch_size, return_datasets=True
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, sampler=eval_sampler, num_workers=4, pin_memory=True)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.epochs * len(train_loader) // args.gradient_accumulation_steps
    lr_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)
    scaler = GradScaler()

    if args.do_adversarial_training:
        fgm = FGM(model)
        if rank == 0:
            print("Adversarial training (FGM) is ENABLED.")
    
    # --- NEW: Initialize a history list to store metrics ---
    training_history = []

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        # --- NEW: Keep track of total loss for the epoch ---
        total_loss = 0.0
        
        data_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(rank != 0))
        
        for i, batch in enumerate(data_iterator):
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
            
            scaler.scale(loss).backward()

            if args.do_adversarial_training:
                fgm.attack(epsilon=args.epsilon)
                with autocast(device_type='cuda', dtype=torch.float16):
                    adv_outputs = model(**inputs, labels=labels)
                    adv_loss = adv_outputs.loss
                    adv_loss = adv_loss / args.gradient_accumulation_steps
                scaler.scale(adv_loss).backward()
                fgm.restore()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # --- NEW: Accumulate the un-normalized loss ---
            total_loss += loss.item() * args.gradient_accumulation_steps

            if rank == 0:
                data_iterator.set_postfix(loss=(loss.item() * args.gradient_accumulation_steps))
        
        # --- NEW: Calculate average training loss for the epoch ---
        avg_train_loss = total_loss / len(train_loader)
        
        eval_results = evaluate(model, eval_loader, device)
        dist.barrier(device_ids=[local_rank])
        
        if rank == 0:
            print(f"\n--- Evaluating after Epoch {epoch+1} ---")
            val_accuracy = eval_results.get('accuracy', 0)
            val_f1 = eval_results.get('f1', 0)
            print(f"Average Training Loss: {avg_train_loss:.4f}")
            print(f"Validation Results: Accuracy={val_accuracy:.4f}, F1={val_f1:.4f}")
            
            # --- NEW: Append metrics for the current epoch to the history ---
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1
            })

    # --- Final Model Saving and Visualization (only on main process) ---
    if rank == 0:
        if args.output_dir:
            print(f"\n--- Saving final model and metrics to {args.output_dir} ---")
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Save the model and tokenizer
            model.module.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print("Model and tokenizer saved successfully.")

            # --- Save metrics history to a JSON file ---
            history_path = os.path.join(args.output_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=4)
            print(f"Training history saved to {history_path}")

            # --- Call the visualizer to plot and save metrics ---
            if training_history: # Ensure history is not empty
                plot_and_save_metrics(training_history, args.output_dir)
    
    cleanup_ddp()

if __name__ == "__main__":
    main()