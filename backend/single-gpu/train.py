# train.py

# --- 1. Import Libraries ---
import torch
import argparse
import os
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

# Import custom modules
from data_loader import get_nli_dataloader, get_spatial_dataloader
from fgm_attack import FGM
from evaluator import evaluate, evaluate_adversarial

# --- Main Training Function ---
def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train a text classification model with optional adversarial training.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the folder containing the pretrained model and tokenizer.")
    parser.add_argument("--dataset_name", type=str, default="snli", choices=["snli", "mnli", "spatial"], help="Name of the dataset to use.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--do_adversarial_training", action='store_true', help="If set, perform FGM adversarial training.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Perturbation size (epsilon) for FGM attack.")
    parser.add_argument("--output_dir", type=str, default="./trained_model", help="Directory where the final model will be saved.")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU to use.")
    args = parser.parse_args()

    # --- Device Setup ---
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model and Tokenizer ---
    num_labels = 3 if args.dataset_name in ["snli", "mnli"] else 2
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels).to(device)

    # --- Load Data ---
    if args.dataset_name in ["snli", "mnli"]:
        train_loader, eval_loader, test_loader = get_nli_dataloader(args.dataset_name, tokenizer, args.batch_size)
    else: # spatial dataset
        # This is a simplified setup for demonstration purposes.
        train_loader = get_spatial_dataloader("mock_spatial_data.csv", tokenizer, args.batch_size)
        eval_loader = train_loader
        test_loader = train_loader
        print("Warning: Using the same mock spatial data for train, eval, and test sets.")

    # --- Optimizer, Scheduler, and Loss Function ---
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.epochs * len(train_loader)
    
    # Use PyTorch's native scheduler for a linear learning rate decay
    lr_scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=num_training_steps
    )

    # --- Adversarial Training Setup ---
    # 注意: evaluator.py中的evaluate_adversarial调用fgm.attack()时没有传递emb_name
    # 这要求你的FGM类在初始化时已经知道了要攻击的嵌入层名称。
    if args.do_adversarial_training:
        fgm = FGM(model) # 假设FGM类内部处理了'word_embeddings'
        print("Adversarial training (FGM) is ENABLED.")
    
    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train() # Set model to training mode
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            # Separate labels and move data to the device
            labels = batch.pop('labels').to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # --- Standard Forward and Backward Pass ---
            outputs = model(**inputs, labels=labels) # Pass labels for loss calculation
            loss = outputs.loss
            loss.backward() # Compute gradients for the clean batch

            # --- Adversarial Training Step (if enabled) ---
            if args.do_adversarial_training:
                # 1. Apply FGM attack to perturb word embeddings
                fgm.attack(epsilon=args.epsilon) # 调用攻击
                # 2. Forward pass with the perturbed embeddings
                adv_outputs = model(**inputs, labels=labels) # 传递相同的输入和标签
                # 3. Calculate loss on the adversarial sample
                adv_loss = adv_outputs.loss
                # 4. Backpropagate the adversarial loss (gradients accumulate)
                adv_loss.backward()
                # 5. Restore original embeddings before the next step
                fgm.restore() # 恢复嵌入

            # --- Optimizer and Scheduler Step ---
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item())

        # --- Evaluation after each epoch ---
        print(f"\n--- Evaluating after Epoch {epoch+1} ---")
        eval_results = evaluate(model, eval_loader, device)
        print(f"Validation Results: Accuracy={eval_results['accuracy']:.4f}, F1={eval_results['f1']:.4f}")

    # Final Evaluation on Test Set ---
    print("\n--- Final Evaluation on Test Set ---")
    print("--- Evaluating on CLEAN test data ---")
    test_results_clean = evaluate(model, test_loader, device)
    print(f"Clean Test Results: Accuracy={test_results_clean['accuracy']:.4f}, F1={test_results_clean['f1']:.4f}")

    if args.do_adversarial_training:
        print("\n--- Evaluating on ADVERSARIAL test data ---")
        test_results_adv = evaluate_adversarial(model, test_loader, device, fgm, args.epsilon)
        print(f"Adversarial Test Results: Robust Accuracy={test_results_adv['robust_accuracy']:.4f}, Adversarial Success Rate={test_results_adv['adversarial_success_rate']:.4f}")
    
    # Save the Final Model
    if args.output_dir:
        print(f"\n--- Saving final model to {args.output_dir} ---")
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("Model and tokenizer saved successfully.")

if __name__ == "__main__":
    main()
    