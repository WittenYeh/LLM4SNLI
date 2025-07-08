# evaluator.py

import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

def evaluate(model, dataloader, device):
    """
    在干净数据集上评估模型
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 将batch中的数据移到指定设备
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
    评估模型在对抗攻击下的鲁棒性
    """
    model.eval()
    
    total_samples = 0
    originally_correct = 0
    flipped_to_wrong = 0
    
    for batch in tqdm(dataloader, desc="Adversarial Evaluating"):
        inputs = {k: v.to(device) for k, v in batch.items()}
        labels = inputs.pop('labels')
        
        # 1. 获得原始预测
        with torch.no_grad():
            clean_outputs = model(**inputs)
            clean_preds = torch.argmax(clean_outputs.logits, dim=-1)
        
        # 2. 生成对抗扰动 (需要计算梯度)
        # 为了生成攻击，我们需要计算损失和梯度
        model.zero_grad()
        # 确保label在模型输入中
        inputs['labels'] = labels
        loss = model(**inputs).loss
        loss.backward()
        
        # 3. 应用攻击并获得对抗预测
        fgm_attack.attack(epsilon=epsilon)
        with torch.no_grad():
            adv_outputs = model(**inputs)
            adv_preds = torch.argmax(adv_outputs.logits, dim=-1)
        fgm_attack.restore()
        
        # 4. 统计结果
        correct_mask = (clean_preds == labels)
        adv_correct_mask = (adv_preds == labels)
        
        total_samples += len(labels)
        originally_correct += correct_mask.sum().item()
        
        # 统计那些本来预测正确，但被攻击后预测错误的样本
        flipped_mask = correct_mask & (~adv_correct_mask)
        flipped_to_wrong += flipped_mask.sum().item()

    if originally_correct == 0:
        return {"adversarial_success_rate": 0, "robust_accuracy": 0}

    # 对抗攻击成功率 = 被扳倒的样本数 / 原本正确的样本数
    adv_success_rate = flipped_to_wrong / originally_correct
    # 鲁棒准确率 = 在攻击下仍然正确的样本数 / 总样本数
    robust_accuracy = (originally_correct - flipped_to_wrong) / total_samples

    return {
        "adversarial_success_rate": adv_success_rate,
        "robust_accuracy": robust_accuracy
    }