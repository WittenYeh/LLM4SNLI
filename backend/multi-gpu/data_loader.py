# data_loader.py

# --- 1. Import Libraries ---
# pandas is used for data manipulation, particularly for reading CSV files.
import pandas as pd
# 'load_dataset' is a core function from the Hugging Face 'datasets' library to easily download and cache datasets.
from datasets import load_dataset
# 'DataLoader' is a PyTorch utility to create batches of tensor data. 'TensorDataset' is a utility to wrap tensors into a dataset.
from torch.utils.data import DataLoader, TensorDataset
# PyTorch is the main deep learning framework used.
import torch

# --- 2. NLI DataLoader Function ---
def get_nli_dataloader(dataset_name, tokenizer, batch_size=32, max_length=128, return_datasets=False):
    """
    Creates and returns PyTorch DataLoaders or Datasets for the SNLI or MNLI dataset.

    This function handles loading the dataset from the Hugging Face Hub, preprocessing the text data,
    tokenizing it, and wrapping it in DataLoader objects for training, validation, and testing.

    Args:
        dataset_name (str): The name of the dataset, either 'snli' or 'mnli'.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for processing text.
        batch_size (int): The number of samples per batch.
        max_length (int): The maximum sequence length for tokenization.
        return_datasets (bool): If True, returns raw Dataset objects instead of DataLoaders.
    
    Returns:
        tuple: A tuple containing train, evaluation, and test DataLoaders or Datasets.
    """
    
    # Define a preprocessing function that will be applied to the dataset.
    def preprocess_function(examples):
        """Tokenizes premise and hypothesis pairs."""
        return tokenizer(examples['premise'], examples['hypothesis'],
                        truncation=True, padding='max_length', max_length=max_length)

    # --- Load Dataset from Hugging Face Hub ---
    if dataset_name == 'snli':
        dataset = load_dataset('snli')
    elif dataset_name == 'mnli':
        dataset = load_dataset('glue', 'mnli')
    else:
        raise ValueError("Dataset name must be either 'snli' or 'mnli'")

    # --- Preprocess and Clean the Dataset ---
    dataset = dataset.filter(lambda example: example['label'] != -1)
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    # --- Remove Unnecessary Columns ---
    tokenized_datasets = tokenized_datasets.remove_columns(['premise', 'hypothesis'])
    if 'premise_binary_parse' in tokenized_datasets['train'].column_names:
        tokenized_datasets = tokenized_datasets.remove_columns([
            'premise_binary_parse', 'premise_parse', 'hypothesis_binary_parse', 'hypothesis_parse'
        ])
    if 'idx' in tokenized_datasets['train'].column_names:
        tokenized_datasets = tokenized_datasets.remove_columns(['idx'])

    # Rename the 'label' column to 'labels'.
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Set the format of the dataset to PyTorch tensors.
    tokenized_datasets.set_format("torch")

    # --- Return either Datasets or DataLoaders ---
    val_key = 'validation_matched' if dataset_name == 'mnli' else 'validation'
    test_key = 'test_matched' if dataset_name == 'mnli' else 'test'
    
    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets[val_key]
    test_dataset = tokenized_datasets[test_key]

    if return_datasets:
        return train_dataset, eval_dataset, test_dataset

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, eval_dataloader, test_dataloader

# --- 3. Custom Spatial DataLoader Function ---

def get_spatial_dataloader(file_path, tokenizer, batch_size=32, max_length=128):
    """
    Creates a DataLoader for a custom spatial reasoning dataset from a CSV file.

    Note: This function includes a mock data creation step for demonstration. In a real
    scenario, you would load your pre-existing data file. This function is designed
    for a simple binary classification task.

    Args:
        file_path (str): The path to the CSV data file.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for processing text.
        batch_size (int): The number of samples per batch.
        max_length (int): The maximum sequence length for tokenization.

    Returns:
        DataLoader: A DataLoader for the custom dataset.
    """
    # ---- Mock Data Creation (for demonstration purposes) ----
    # This block creates a dummy CSV file if it doesn't exist.
    # In a real application, you would provide your own CSV file.
    mock_data = {
        'description': [
            "The red block is to the left of the blue sphere.",
            "Object A is 5 meters behind Object B.",
            "The cube is not on top of the pyramid."
        ],
        'labels': [1, 1, 0] # Example labels: 1 for True, 0 for False
    }
    mock_df = pd.DataFrame(mock_data)
    mock_df.to_csv(file_path, index=False)
    # ---- End of Mock Data Creation ----

    # Load the data from the specified CSV file into a pandas DataFrame.
    df = pd.read_csv(file_path)
    # Extract the text descriptions and labels into Python lists.
    texts = df['description'].tolist()
    labels = df['labels'].tolist()

    # Tokenize the list of texts. 'return_tensors='pt'' specifies that the output
    # should be PyTorch tensors.
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    
    # Create a TensorDataset. This is a PyTorch-native way to create a dataset from
    # tensors. It aligns the tensors so that the i-th element of each tensor corresponds
    # to the i-th sample.
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(labels))
    
    # Wrap the dataset in a DataLoader.
    # Note: For a real use case, you should split your data into train, validation, and test sets
    # before creating separate DataLoaders for each.
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    
    return dataloader