import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model_classification import get_model
import matplotlib.pyplot as plt
from data import load_and_preprocess_dataset
from sklearn.preprocessing import LabelEncoder

# Basic configurations - Modified
HEIGHT, WIDTH = 224, 224
CHANNELS = 3
NUM_CLASSES = 5
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.001 
WEIGHT_DECAY = 1e-4  

def load_data():
    """Load and split the preprocessed dataset"""
    # Load dataset using the function from data.py
    images, _, labels = load_and_preprocess_dataset(image_size=(HEIGHT, WIDTH))
    
    # Normalize images to [0,1] and standardize
    images = images.astype(np.float32) / 255.0
    mean = np.mean(images, axis=(0,1,2))
    std = np.std(images, axis=(0,1,2))
    images = (images - mean) / (std + 1e-7)
    
    # Convert string labels to numeric using LabelEncoder
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Split into train and validation (80-20 split)
    np.random.seed(42)  # Added for reproducibility
    indices = np.random.permutation(len(images))
    split_idx = int(0.8 * len(images))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_images = images[train_indices]
    train_labels = labels_encoded[train_indices]
    val_images = images[val_indices]
    val_labels = labels_encoded[val_indices]
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    print(f"Classes: {label_encoder.classes_}")
    print(f"Label distribution: {np.bincount(labels_encoded)}")
    
    return train_images, train_labels, val_images, val_labels

if __name__ == "__main__":
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load and preprocess data
    print("Loading data...")
    train_images, train_labels, val_images, val_labels = load_data()

    # Convert to PyTorch tensors
    train_images = torch.FloatTensor(train_images).permute(0, 3, 1, 2)  # NHWC -> NCHW
    train_labels = torch.LongTensor(train_labels)
    val_images = torch.FloatTensor(val_images).permute(0, 3, 1, 2)
    val_labels = torch.LongTensor(val_labels)

    # Create data loaders
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model with improved optimizer settings
    model = get_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(  # Changed to AdamW
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )

    # Training loop - Add accuracy tracking
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Print progress
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model based on validation accuracy
        if val_acc > max(val_accs[:-1], default=0):
            torch.save(model.state_dict(), 'best_model_classification.pth')
            print(f"Saved new best model with validation accuracy: {val_acc:.2f}%")

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history_classification.png')
    plt.close()