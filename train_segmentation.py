import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model_segmentation import get_model
import matplotlib.pyplot as plt
from data import load_and_preprocess_dataset

# Basic configurations
HEIGHT, WIDTH = 224, 224
CHANNELS = 3
NUM_CLASSES = 5
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.001

def load_data():
    """Load and split the preprocessed dataset"""
    # Load dataset using the function from data.py
    images, masks, labels = load_and_preprocess_dataset(image_size=(HEIGHT, WIDTH))
    
    # Split into train and validation (80-20 split)
    split_idx = int(0.8 * len(images))
    
    train_images = images[:split_idx]
    train_masks = masks[:split_idx]
    val_images = images[split_idx:]
    val_masks = masks[split_idx:]
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    
    return train_images, train_masks, val_images, val_masks

if __name__ == "__main__":
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load and preprocess data
    print("Loading data...")
    train_images, train_masks, val_images, val_masks = load_data()

    # Convert to PyTorch tensors
    train_images = torch.FloatTensor(train_images).permute(0, 3, 1, 2)  # NHWC -> NCHW
    train_masks = torch.LongTensor(train_masks)
    val_images = torch.FloatTensor(val_images).permute(0, 3, 1, 2)
    val_masks = torch.LongTensor(val_masks)

    # Create data loaders
    train_dataset = TensorDataset(train_images, train_masks)
    val_dataset = TensorDataset(val_images, val_masks)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = get_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Print progress
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved new best model with validation loss: {avg_val_loss:.4f}")

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()