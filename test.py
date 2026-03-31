import torch
import numpy as np
from model_segmentation import get_model as get_seg_model
from model_classification import get_model as get_class_model
from data import load_and_preprocess_dataset
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = ['A', 'B', 'C', 'D', 'E']

def load_models():
    """Load both trained models"""
    # Load classification model
    class_model = get_class_model()
    class_model.load_state_dict(torch.load('best_model_classification.pth', map_location=DEVICE))
    class_model.to(DEVICE).eval()
    
    # Load segmentation model
    seg_model = get_seg_model()
    seg_model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    seg_model.to(DEVICE).eval()
    
    return class_model, seg_model

def calculate_segmentation_accuracy(pred_masks, true_masks):
    """Calculate IoU and Dice scores for segmentation"""
    pred_masks = pred_masks > 0.5
    intersection = np.logical_and(true_masks, pred_masks)
    union = np.logical_or(true_masks, pred_masks)
    iou_score = np.mean(np.sum(intersection) / np.sum(union))
    dice_score = 2 * np.sum(intersection) / (np.sum(true_masks) + np.sum(pred_masks))
    return iou_score, dice_score

def plot_confusion_matrix(true_labels, pred_labels, title):
    """Plot confusion matrix"""
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES,
                yticklabels=CLASSES)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt.gcf()

def plot_accuracy_comparison(class_acc, seg_iou, seg_dice):
    """Plot accuracy comparison"""
    plt.figure(figsize=(10, 6))
    metrics = ['Classification Accuracy', 'Segmentation IoU', 'Segmentation Dice']
    values = [class_acc, seg_iou, seg_dice]
    
    plt.bar(metrics, values, color=['blue', 'green', 'orange'])
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    return plt.gcf()

def main():
    print(f"Using device: {DEVICE}")
    
    # Load models
    print("Loading models...")
    class_model, seg_model = load_models()
    
    # Load validation data
    print("Loading validation data...")
    images, masks, labels = load_and_preprocess_dataset()
    
    # Split into validation set (use last 20%)
    val_idx = int(0.8 * len(images))
    val_images = images[val_idx:]
    val_masks = masks[val_idx:]
    val_labels = labels[val_idx:]
    
    # Convert to tensors
    val_images = torch.FloatTensor(val_images).permute(0, 3, 1, 2).to(DEVICE)
    
    # Evaluation
    print("\nEvaluating models...")
    class_predictions = []
    seg_predictions = []
    
    with torch.no_grad():
        for image in val_images:
            # Classification prediction
            class_output = class_model(image.unsqueeze(0))
            class_pred = torch.softmax(class_output, dim=1).argmax(1).item()
            class_predictions.append(class_pred)
            
            # Segmentation prediction
            seg_output = seg_model(image.unsqueeze(0))
            seg_pred = torch.sigmoid(seg_output).cpu().numpy()
            seg_predictions.append(seg_pred[0, 0])
    
    # Calculate metrics
    classification_accuracy = accuracy_score(val_labels, class_predictions)
    iou_score, dice_score = calculate_segmentation_accuracy(
        np.array(seg_predictions), val_masks
    )
    
    # Print results
    print("\nValidation Results:")
    print(f"Classification Accuracy: {classification_accuracy:.3f}")
    print(f"Segmentation IoU Score: {iou_score:.3f}")
    print(f"Segmentation Dice Score: {dice_score:.3f}")
    
    # Plot confusion matrix
    cm_plot = plot_confusion_matrix(
        val_labels, class_predictions,
        'Classification Confusion Matrix'
    )
    cm_plot.savefig('confusion_matrix.png')
    
    # Plot accuracy comparison
    acc_plot = plot_accuracy_comparison(
        classification_accuracy, iou_score, dice_score
    )
    acc_plot.savefig('accuracy_comparison.png')
    
    plt.close('all')
    print("\nPlots saved as 'confusion_matrix.png' and 'accuracy_comparison.png'")

if __name__ == "__main__":
    main()