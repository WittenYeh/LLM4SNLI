# data_loader.py

import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
import torch

def get_nli_dataloader(dataset_name, tokenizer, batch_size=32, max_length=128):
    """
    Create DataLoader for SNLI or MNLI dataset
    """
    def preprocess_function(examples):
        # Hugging Face tokenizer can process both sentences simultaneously
        return tokenizer(examples['premise'], examples['hypothesis'],
                        truncation=True, padding='max_length', max_length=max_length)

    # Load dataset from Hugging Face Hub
    if dataset_name == 'snli':
        dataset = load_dataset('snli')
    elif dataset_name == 'mnli':
        # MNLI has two validation sets: matched and mismatched
        dataset = load_dataset('glue', 'mnli')
    else:
        raise ValueError("Dataset name must be either 'snli' or 'mnli'")

    # Filter out samples with label -1 in SNLI/MNLI
    dataset = dataset.filter(lambda example: example['label'] != -1)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['premise', 'hypothesis', 'idx'])
    if 'premise_binary_parse' in tokenized_datasets['train'].column_names: # SNLI specific
        tokenized_datasets = tokenized_datasets.remove_columns([
            'premise_binary_parse', 'premise_parse', 'hypothesis_binary_parse', 'hypothesis_parse'
        ])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=batch_size)
    
    # For MNLI, we use the matched validation set for evaluation
    val_key = 'validation_matched' if dataset_name == 'mnli' else 'validation'
    eval_dataloader = DataLoader(tokenized_datasets[val_key], batch_size=batch_size)

    test_key = 'test_matched' if dataset_name == 'mnli' else 'test'
    test_dataloader = DataLoader(tokenized_datasets[test_key], batch_size=batch_size)

    return train_dataloader, eval_dataloader, test_dataloader

def get_spatial_dataloader(file_path, tokenizer, batch_size=32, max_length=128):
    """
    Create DataLoader for custom spatial reasoning dataset
    Note: This is an example - you'll need to adapt it for Spatial Eval's actual format
    """
    # ---- Mock data creation ----
    # In practice, you should directly load Spatial Eval or your custom data
    mock_data = {
        'description': [
            "The red block is to the left of the blue sphere.",
            "Object A is 5 meters behind Object B.",
            "The cube is not on top of the pyramid."
        ],
        'labels': [1, 1, 0] # 1 for True, 0 for False
    }
    mock_df = pd.DataFrame(mock_data)
    mock_df.to_csv(file_path, index=False)
    # ---- End of mock ----

    df = pd.read_csv(file_path)
    # Assuming spatial reasoning is a binary classification problem (True/False)
    texts = df['description'].tolist()
    labels = df['labels'].tolist()

    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(labels))
    
    # Here we only create one dataloader as an example - in practice you should split into train/val/test sets
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return dataloader
