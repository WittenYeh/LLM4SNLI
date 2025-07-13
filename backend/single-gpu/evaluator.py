# evaluator.py

import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

def evaluate(model, dataloader, device):
    """
    Evaluate model performance on clean dataset
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch data to specified device
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.pop('labels')

            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return {"accuracy": accuracy, "f1": f1}


def evaluate_adversarial(model, dataloader, device, fgm_attack, epsilon):
    """
    Evaluate model robustness under adversarial attacks
    """
    model.eval()
    
    total_samples = 0
    originally_correct = 0
    flipped_to_wrong = 0
    
    for batch in tqdm(dataloader, desc="Adversarial Evaluating"):
        inputs = {k: v.to(device) for k, v in batch.items()}
        labels = inputs.pop('labels')
        
        # 1. Get original predictions
        with torch.no_grad():
            clean_outputs = model(**inputs)
            clean_preds = torch.argmax(clean_outputs.logits, dim=-1)
        
        # 2. Generate adversarial perturbations (requires gradient calculation)
        # To generate attacks, we need to compute loss and gradients
        model.zero_grad()
        # Ensure labels are included in model inputs
        inputs['labels'] = labels
        loss = model(**inputs).loss
        loss.backward()
        
        # 3. Apply attack and get adversarial predictions
        fgm_attack.attack(epsilon=epsilon)
        with torch.no_grad():
            adv_outputs = model(**inputs)
            adv_preds = torch.argmax(adv_outputs.logits, dim=-1)
        fgm_attack.restore()
        
        # 4. Collect statistics
        correct_mask = (clean_preds == labels)
        adv_correct_mask = (adv_preds == labels)
        
        total_samples += len(labels)
        originally_correct += correct_mask.sum().item()
        
        # Count samples that were originally correct but became wrong after attack
        flipped_mask = correct_mask & (~adv_correct_mask)
        flipped_to_wrong += flipped_mask.sum().item()

    if originally_correct == 0:
        return {"adversarial_success_rate": 0, "robust_accuracy": 0}

    # Adversarial success rate = flipped samples / originally correct samples
    adv_success_rate = flipped_to_wrong / originally_correct
    # Robust accuracy = samples that remain correct under attack / total samples
    robust_accuracy = (originally_correct - flipped_to_wrong) / total_samples

    return {
        "adversarial_success_rate": adv_success_rate,
        "robust_accuracy": robust_accuracy
    }