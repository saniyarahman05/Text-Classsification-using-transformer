import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AdamW
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from data_preparation import prepare_data, create_dataloaders

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(val_loader), accuracy, f1

def train_model(data_path, num_epochs=5, batch_size=32, learning_rate=2e-5):
    # Initialize wandb
    wandb.init(project="transformer-training", name="experiment-1")
    
    # Prepare data
    train_dataset, val_dataset, test_dataset, tokenizer = prepare_data(data_path)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2  # Modify this based on your number of classes
    )
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate
        val_loss, val_accuracy, val_f1 = evaluate(model, val_loader, device)
        
        # Log metrics
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1
        })
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        print(f"Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
    
    # Final evaluation on test set
    test_loss, test_accuracy, test_f1 = evaluate(model, test_loader, device)
    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    
    wandb.finish()

if __name__ == "__main__":
    data_path = "sample_data.csv"
    train_model(data_path) 