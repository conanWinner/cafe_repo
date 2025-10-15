#!/usr/bin/env python3
"""
Evaluation Script: Evaluate trained model and generate reports
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
from pathlib import Path
from tqdm import tqdm

# Import from train.py
from train import CoffeeLeafClassifier, CoffeeLeafDataset, CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path, num_classes=6):
    """Load trained model from checkpoint"""
    model = CoffeeLeafClassifier(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint

def evaluate_model(model, dataloader, classes):
    """Evaluate model and return predictions and labels"""
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Plot percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=classes, yticklabels=classes,
                ax=axes[1], cbar_kws={'label': 'Percentage (%)'})
    axes[1].set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {output_path}")
    plt.close()

def plot_per_class_metrics(report_dict, classes, output_path):
    """Plot per-class precision, recall, f1-score"""
    metrics = ['precision', 'recall', 'f1-score']
    data = {metric: [] for metric in metrics}
    
    for cls in classes:
        if cls in report_dict:
            for metric in metrics:
                data[metric].append(report_dict[cls][metric] * 100)
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, metric in enumerate(metrics):
        offset = width * (i - 1)
        bars = ax.bar(x + offset, data[metric], width, label=metric.capitalize())
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Per-class metrics plot saved to {output_path}")
    plt.close()

def plot_top_k_accuracy(y_true, y_probs, classes, output_path, max_k=5):
    """Plot top-k accuracy"""
    k_values = range(1, min(max_k + 1, len(classes) + 1))
    accuracies = []
    
    for k in k_values:
        # Get top-k predictions
        top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
        correct = sum([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
        acc = correct / len(y_true) * 100
        accuracies.append(acc)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(k_values, accuracies, color='steelblue', edgecolor='navy', linewidth=1.5)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('k (Top-k Predictions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top-k Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(k_values)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Top-k accuracy plot saved to {output_path}")
    plt.close()

def generate_report(checkpoint_path, output_dir=None):
    """Generate comprehensive evaluation report"""
    
    checkpoint_path = Path(checkpoint_path)
    
    # Create output directory
    if output_dir is None:
        output_dir = checkpoint_path.parent.parent / 'evaluation'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("📊 COFFEE LEAF DISEASE CLASSIFICATION - EVALUATION")
    print("="*70)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}\n")
    
    # Load model
    print("Loading model...")
    model, checkpoint = load_model(checkpoint_path)
    
    if 'val_acc' in checkpoint:
        print(f"  Validation Accuracy (from training): {checkpoint['val_acc']:.2f}%")
    if 'epoch' in checkpoint:
        print(f"  Trained Epochs: {checkpoint['epoch']}")
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = CoffeeLeafDataset(
        CONFIG['data_root'], 
        split='valid', 
        transform=val_transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred, y_true, y_probs = evaluate_model(model, val_loader, CONFIG['classes'])
    
    # Calculate metrics
    accuracy = (y_pred == y_true).mean() * 100
    
    print(f"\n{'='*70}")
    print("📈 EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print(f"Total Samples: {len(y_true)}")
    print(f"Correct Predictions: {(y_pred == y_true).sum()}")
    print(f"Wrong Predictions: {(y_pred != y_true).sum()}")
    
    # Classification report
    print(f"\n{'-'*70}")
    print("CLASSIFICATION REPORT")
    print(f"{'-'*70}")
    report = classification_report(y_true, y_pred, target_names=CONFIG['classes'], digits=4)
    print(report)
    
    # Save text report
    report_dict = classification_report(y_true, y_pred, target_names=CONFIG['classes'], 
                                       output_dict=True, digits=4)
    
    report_text_path = output_dir / 'classification_report.txt'
    with open(report_text_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("COFFEE LEAF DISEASE CLASSIFICATION - EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"Total Samples: {len(y_true)}\n\n")
        f.write("-"*70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("-"*70 + "\n")
        f.write(report)
    
    print(f"\n✓ Text report saved to {report_text_path}")
    
    # Save JSON report
    report_json_path = output_dir / 'classification_report.json'
    with open(report_json_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    print(f"✓ JSON report saved to {report_json_path}")
    
    # Plot confusion matrix
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(y_true, y_pred, CONFIG['classes'], cm_path)
    
    # Plot per-class metrics
    metrics_path = output_dir / 'per_class_metrics.png'
    plot_per_class_metrics(report_dict, CONFIG['classes'], metrics_path)
    
    # Plot top-k accuracy
    topk_path = output_dir / 'top_k_accuracy.png'
    plot_top_k_accuracy(y_true, y_probs, CONFIG['classes'], topk_path)
    
    # Save predictions
    predictions_path = output_dir / 'predictions.json'
    predictions_data = {
        'predictions': [
            {
                'true_label': CONFIG['classes'][true],
                'predicted_label': CONFIG['classes'][pred],
                'correct': bool(true == pred),
                'probabilities': {
                    cls: float(prob) 
                    for cls, prob in zip(CONFIG['classes'], probs)
                }
            }
            for true, pred, probs in zip(y_true, y_pred, y_probs)
        ]
    }
    
    with open(predictions_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    print(f"✓ Predictions saved to {predictions_path}")
    
    print(f"\n{'='*70}")
    print("✅ EVALUATION COMPLETED!")
    print(f"{'='*70}")
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - {report_text_path.name}")
    print(f"  - {report_json_path.name}")
    print(f"  - {cm_path.name}")
    print(f"  - {metrics_path.name}")
    print(f"  - {topk_path.name}")
    print(f"  - {predictions_path.name}")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <checkpoint_path> [output_dir]")
        print("\nExample:")
        print("  python evaluate.py outputs/train_20241015_120000/checkpoints/best_model.pth")
        print("  python evaluate.py outputs/train_20241015_120000/checkpoints/best_model.pth evaluation_results/")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    generate_report(checkpoint_path, output_dir)
