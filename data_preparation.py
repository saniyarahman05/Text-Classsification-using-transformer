import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_data(data_path, test_size=0.2, val_size=0.1, random_state=42):
    """
    Prepare the dataset for training
    
    Args:
        data_path (str): Path to the dataset file
        test_size (float): Proportion of dataset to include in test split
        val_size (float): Proportion of dataset to include in validation split
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, tokenizer)
    """
    # Load your dataset here
    # This is a placeholder - you'll need to modify this based on your actual data format
    df = pd.read_csv(data_path)
    
    # Assuming your data has 'text' and 'label' columns
    texts = df['text'].values
    labels = df['label'].values
    
    # Split the data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state
    )
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=val_size, random_state=random_state
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer)
    
    return train_dataset, val_dataset, test_dataset, tokenizer

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """
    Create DataLoaders for training, validation, and testing
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size (int): Batch size for training
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader 