# train.py

# --- 1. Import Libraries ---
import torch
import argparse
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score

# Import custom modules
from data_loader import get_nli_dataloader, get_spatial_dataloader
from fgm_attack import FGM

# --- 2. Evaluation Functions ---

def evaluate_native(model, dataloader, device):
    """
    Evaluates the model on a given dataset using native PyTorch.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation data.
        device (torch.device): The device (CPU or GPU) to run evaluation on.

    Returns:
        dict: A dictionary containing 'accuracy' and 'f1' score.
    """
    # Set the model to evaluation mode (disables dropout, etc.)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # Disable gradient calculation to save memory and speed up inference
    with torch.no_grad():
        for batch in dataloader:
            # Separate labels from the input batch, as the model won't use them for inference
            labels = batch.pop('labels')
            
            # Move inputs and labels to the designated device
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)

            # Get model outputs. Since we don't pass labels, it returns logits.
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get the predicted class index by finding the max logit score
            preds = torch.argmax(logits, dim=-1)
            
            # Collect predictions and labels from all batches
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate metrics using the collected data
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return {"accuracy": accuracy, "f1": f1}

def evaluate_adversarial_native(model, dataloader, device, fgm, epsilon):
    """
    Evaluates the model's robustness against FGM adversarial attacks.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the test data.
        device (torch.device): The device (CPU or GPU).
        fgm (FGM): The FGM attack object.
        epsilon (float): The perturbation size for the attack.

    Returns:
        dict: A dictionary containing robust accuracy and attack success rate.
    """
    model.eval()
    
    correct_predictions_under_attack = 0
    total_samples = 0
    successful_attacks = 0
    total_originally_correct = 0

    for batch in tqdm(dataloader, desc="Adversarial Evaluation"):
        labels = batch.pop('labels')
        inputs = {k: v.to(device) for k, v in batch.items()}
        labels = labels.to(device)

        # 1. Get the model's prediction on the original, clean data
        with torch.no_grad():
            clean_outputs = model(**inputs)
            clean_preds = torch.argmax(clean_outputs.logits, dim=-1)
        
        # 2. Generate adversarial perturbation
        model.zero_grad()
        # Perform a forward pass *with labels* to get the loss, needed for gradient calculation
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()  # Compute gradients w.r.t. embeddings
        fgm.attack(epsilon=epsilon, emb_name='word_embeddings') # Apply the attack

        # 3. Get the model's prediction on the perturbed (adversarial) data
        with torch.no_grad():
            adv_outputs = model(**inputs)
            adv_preds = torch.argmax(adv_outputs.logits, dim=-1)
        
        # 4. IMPORTANT: Restore the original embeddings for the next batch
        fgm.restore(emb_name='word_embeddings')

        # 5. Calculate metrics for this batch
        originally_correct_mask = (clean_preds == labels)
        attacked_correct_mask = (adv_preds == labels)
        
        # Robust Accuracy: samples that are still correct after the attack.
        correct_predictions_under_attack += attacked_correct_mask.sum().item()
        
        # Adversarial Success Rate: samples that were correct but became incorrect after the attack.
        successful_attacks += (originally_correct_mask & ~attacked_correct_mask).sum().item()
        
        total_originally_correct += originally_correct_mask.sum().item()
        total_samples += len(labels)

    robust_accuracy = correct_predictions_under_attack / total_samples if total_samples > 0 else 0.0
    adversarial_success_rate = successful_attacks / total_originally_correct if total_originally_correct > 0 else 0.0

    return {
        "robust_accuracy": robust_accuracy,
        "adversarial_success_rate": adversarial_success_rate
    }


# --- 3. Main Training Function ---
def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train a text classification model with optional adversarial training.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Name of the pretrained model from Hugging Face.")
    parser.add_argument("--dataset_name", type=str, default="snli", choices=["snli", "mnli", "spatial"], help="Name of the dataset to use.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--do_adversarial_training", action='store_true', help="If set, perform FGM adversarial training.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Perturbation size (epsilon) for FGM attack.")
    args = parser.parse_args()

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model and Tokenizer ---
    num_labels = 3 if args.dataset_name in ["snli", "mnli"] else 2
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels).to(device)

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
    if args.do_adversarial_training:
        fgm = FGM(model)
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
            outputs = model(**inputs)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward() # Compute gradients for the clean batch

            # --- Adversarial Training Step (if enabled) ---
            if args.do_adversarial_training:
                # 1. Apply FGM attack to perturb word embeddings
                fgm.attack(epsilon=args.epsilon, emb_name='word_embeddings')
                # 2. Forward pass with the perturbed embeddings
                adv_outputs = model(**inputs)
                adv_logits = adv_outputs.logits
                # 3. Calculate loss on the adversarial sample
                adv_loss = loss_fn(adv_logits, labels)
                # 4. Backpropagate the adversarial loss (gradients accumulate)
                adv_loss.backward()
                # 5. Restore original embeddings before the next step
                fgm.restore(emb_name='word_embeddings')

            # --- Optimizer and Scheduler Step ---
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item())

        # --- Evaluation after each epoch ---
        print(f"\n--- Evaluating after Epoch {epoch+1} ---")
        eval_results = evaluate_native(model, eval_loader, device)
        print(f"Validation Results: Accuracy={eval_results['accuracy']:.4f}, F1={eval_results['f1']:.4f}")

    # --- Final Evaluation on Test Set ---
    print("\n--- Final Evaluation on Test Set ---")
    print("--- Evaluating on CLEAN test data ---")
    test_results_clean = evaluate_native(model, test_loader, device)
    print(f"Clean Test Results: Accuracy={test_results_clean['accuracy']:.4f}, F1={test_results_clean['f1']:.4f}")

    if args.do_adversarial_training:
        print("\n--- Evaluating on ADVERSARIAL test data ---")
        test_results_adv = evaluate_adversarial_native(model, test_loader, device, fgm, args.epsilon)
        print(f"Adversarial Test Results: Robust Accuracy={test_results_adv['robust_accuracy']:.4f}, Adversarial Success Rate={test_results_adv['adversarial_success_rate']:.4f}")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Note: scikit-learn is required for evaluation. If not installed, run:
    # pip install scikit-learn
    main()