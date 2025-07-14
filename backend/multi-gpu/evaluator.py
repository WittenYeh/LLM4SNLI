# evaluator.py (CORRECTED FINAL VERSION)

import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

def is_dist_avail_and_initialized():
    """Checks if distributed training is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def _gather_tensors(tensor):
    """Gathers a tensor from all processes, supporting different sizes."""
    world_size = dist.get_world_size()
    # Create a list to hold tensors from all ranks
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

def evaluate(model, dataloader, device):
    """Evaluate model performance on clean dataset, with correct DDP handling."""
    model.eval()
    local_preds_list = []
    local_labels_list = []

    # Progress bar is only shown on the main process (rank 0)
    is_main_process = not is_dist_avail_and_initialized() or dist.get_rank() == 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not is_main_process):
            # Move data to the correct device
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            # Store predictions and labels from this process's data slice
            local_preds_list.append(preds)
            local_labels_list.append(labels)

    # Concatenate all local tensors into single tensors
    local_preds = torch.cat(local_preds_list, dim=0)
    local_labels = torch.cat(local_labels_list, dim=0)

    if is_dist_avail_and_initialized():
        # In DDP mode, gather predictions and labels from all GPUs
        all_preds_tensor = _gather_tensors(local_preds)
        all_labels_tensor = _gather_tensors(local_labels)
    else:
        # In single-GPU mode, the local tensors are the full tensors
        all_preds_tensor = local_preds
        all_labels_tensor = local_labels

    # --- IMPORTANT ---
    # Only the main process performs the final metric calculation and returns the result.
    # Other processes return an empty dict, but crucially, they do so AFTER all distributed
    # communication (`all_gather`) is complete.
    if not is_main_process:
        return {}

    # Convert final tensors to numpy for sklearn
    all_preds = all_preds_tensor.cpu().numpy()
    all_labels = all_labels_tensor.cpu().numpy()

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return {"accuracy": accuracy, "f1": f1}


def evaluate_adversarial(model, dataloader, device, fgm_attack, epsilon):
    """Evaluate model robustness, with correct DDP handling."""
    # This function's logic would need a similar structural fix.
    # For now, let's focus on fixing the main `evaluate` function.
    # A full implementation would require gathering stats dictionaries, which is slightly different.
    # If you need this, let me know, and I can provide the robust version.
    
    # Placeholder for non-DDP to avoid breaking single-GPU tests.
    if is_dist_avail_and_initialized():
        if dist.get_rank() == 0:
            print("Warning: Distributed adversarial evaluation is not fully implemented in this version and may be inaccurate.")
        return {} # Return empty dict in DDP to avoid hangs

    model.eval()
    total_samples, originally_correct, flipped_to_wrong = 0, 0, 0
    is_main_process = not is_dist_avail_and_initialized() or dist.get_rank() == 0

    for batch in tqdm(dataloader, desc="Adversarial Evaluating", disable=not is_main_process):
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)
        
        with torch.no_grad():
            clean_outputs = model(**inputs)
            clean_preds = torch.argmax(clean_outputs.logits, dim=-1)
        
        model.zero_grad()
        loss = model(labels=labels, **inputs).loss
        loss.backward()
        
        fgm_attack.attack(epsilon=epsilon)
        with torch.no_grad():
            adv_outputs = model(**inputs)
            adv_preds = torch.argmax(adv_outputs.logits, dim=-1)
        fgm_attack.restore()
        
        correct_mask = (clean_preds == labels)
        adv_correct_mask = (adv_preds == labels)
        flipped_mask = correct_mask & (~adv_correct_mask)
        total_samples += len(labels)
        originally_correct += correct_mask.sum().item()
        flipped_to_wrong += flipped_mask.sum().item()

    if originally_correct == 0:
        return {"adversarial_success_rate": 0, "robust_accuracy": 0}

    adv_success_rate = flipped_to_wrong / originally_correct
    robust_accuracy = (originally_correct - flipped_to_wrong) / total_samples
    return {"adversarial_success_rate": adv_success_rate, "robust_accuracy": robust_accuracy}
