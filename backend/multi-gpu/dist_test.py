# dist_test.py (MODIFIED - with results file output)

import torch
import argparse
import os
import json # Import json for structured output
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

# Import your custom, DDP-aware modules
from data_loader import get_nli_dataloader
from fgm_attack import FGM
from evaluator import evaluate, evaluate_adversarial

def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def main():
    # --- DDP Setup ---
    setup_ddp()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Distributed testing of a trained text classification model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the folder containing the TRAINED model and tokenizer.")
    parser.add_argument("--dataset_name", type=str, default="snli", choices=["snli", "mnli"], help="Name of the NLI dataset to use for testing.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size PER GPU for evaluation.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Perturbation size (epsilon) for FGM attack.")
    # --- NEW ARGUMENT for saving results ---
    parser.add_argument("--results_file", type=str, default=None, help="Optional path to a file to save the test results.")
    args = parser.parse_args()

    # --- Device Setup ---
    device = torch.device(f"cuda:{local_rank}")
    if rank == 0:
        print(f"Starting distributed testing on {world_size} GPUs.")

    # --- Load Trained Model and Tokenizer ---
    if rank == 0:
        print(f"Loading model and tokenizer from {args.model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    model.eval()

    # --- Load and Distribute Test Data ---
    if rank == 0:
        print(f"Loading test data for {args.dataset_name}...")
    
    _, _, test_dataset = get_nli_dataloader(
        args.dataset_name, tokenizer, args.batch_size, return_datasets=True
    )
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=4, pin_memory=True)

    # --- Initialize FGM for Adversarial Evaluation ---
    fgm = FGM(model)

    # --- Run Evaluations ---
    clean_results = evaluate(model, test_loader, device)
    adversarial_results = evaluate_adversarial(model, test_loader, device, fgm, args.epsilon)

    # --- Report and Save Final Metrics (only on main process) ---
    dist.barrier(device_ids=[local_rank])
    
    if rank == 0:
        # Prepare the results dictionary for structured output
        final_results = {
            "model_path": args.model_path,
            "dataset": args.dataset_name,
            "accuracy": clean_results.get('accuracy', 0),
            "f1_score_macro": clean_results.get('f1', 0),
            "adversarial_success_rate": adversarial_results.get('adversarial_success_rate', 0)
        }
        
        # Format the output string for both console and file
        output_string = (
            f"\n{'='*50}\n"
            f"{'FINAL TEST RESULTS'.center(50)}\n"
            f"{'='*50}\n"
            f"  Model Path:               {final_results['model_path']}\n"
            f"  Dataset:                  {final_results['dataset']}\n"
            f"  --------------------------------------------------\n"
            f"  Accuracy:                 {final_results['accuracy']:.4f}\n"
            f"  F1-Score (Macro):         {final_results['f1_score_macro']:.4f}\n"
            f"  Adversarial Success Rate: {final_results['adversarial_success_rate']:.4f}\n"
            f"{'='*50}"
        )

        # Print the results to the console
        print(output_string)

        # If a results file path is provided, save the results
        if args.results_file:
            # Ensure the directory for the results file exists
            results_dir = os.path.dirname(args.results_file)
            if results_dir:
                os.makedirs(results_dir, exist_ok=True)
            
            # Write the results to the file. We'll use a JSON format for easy parsing later.
            try:
                with open(args.results_file, 'w') as f:
                    json.dump(final_results, f, indent=4)
                print(f"\nResults successfully saved to: {args.results_file}")
            except Exception as e:
                print(f"\nError: Could not save results to file. Reason: {e}")

    # --- Cleanup ---
    cleanup_ddp()

if __name__ == "__main__":
    main()