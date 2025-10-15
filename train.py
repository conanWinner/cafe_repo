#!/usr/bin/env python3
"""
Fine-tuning Script: Coffee Leaf Disease Classification
Load pretrained checkpoint and train on ft_data
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import Counter

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###################################
#   Configuration
###################################
CONFIG = {
    'data_root': 'data/ft_data',
    'checkpoint_path': 'ckpt/resnet50_ssl_141025.pt',
    'output_dir': 'outputs',
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'num_classes': 6,
    'image_size': 224,
    'num_workers': 4,
    'save_every': 5,  # Save checkpoint every N epochs
    'classes': ['Cercospora', 'Corticium', 'mealy', 'Miner', 'Phoma', 'Rust'],
    # Class imbalance handling
    'use_class_weights': True,  # Use weighted loss
    'use_weighted_sampler': True,  # Use weighted random sampler
    'weight_scheme': 'inverse',  # 'inverse' or 'effective' or 'sqrt_inverse'
}

###################################
#   Helper Functions
###################################
def calculate_class_weights(labels, num_classes, scheme='inverse'):
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        labels: List of all labels in dataset
        num_classes: Number of classes
        scheme: 'inverse', 'sqrt_inverse', or 'effective'
    
    Returns:
        torch.Tensor: Class weights
    """
    # Count samples per class
    class_counts = Counter(labels)
    counts = torch.tensor([class_counts[i] for i in range(num_classes)], dtype=torch.float32)
    
    if scheme == 'inverse':
        # Inverse frequency: weight = 1 / count
        weights = 1.0 / counts
    elif scheme == 'sqrt_inverse':
        # Square root of inverse frequency
        weights = 1.0 / torch.sqrt(counts)
    elif scheme == 'effective':
        # Effective number of samples: (1 - beta^n) / (1 - beta)
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown weight scheme: {scheme}")
    
    # Normalize weights so they sum to num_classes
    weights = weights / weights.sum() * num_classes
    
    return weights

###################################
#   Dataset Class
###################################
class CoffeeLeafDataset(Dataset):
    """Coffee Leaf Disease Dataset"""
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.classes = CONFIG['classes']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist!")
                continue
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"{split.upper()} set: {len(self.samples)} images")
        
        # Print class distribution
        class_counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_counts[self.classes[label]] += 1
        
        # Calculate imbalance ratio
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"  Class distribution:")
        for cls, count in class_counts.items():
            percentage = 100.0 * count / len(self.samples)
            print(f"    - {cls:12s}: {count:4d} images ({percentage:5.1f}%)")
        
        if imbalance_ratio > 2.0:
            print(f"  ⚠️  Imbalance detected! Ratio: {imbalance_ratio:.2f}:1")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_labels(self):
        """Return list of all labels in dataset"""
        return [label for _, label in self.samples]

###################################
#   Model Architecture
###################################
class CoffeeLeafClassifier(nn.Module):
    """ResNet50 based classifier with pretrained encoder"""
    def __init__(self, num_classes=6, pretrained_path=None):
        super(CoffeeLeafClassifier, self).__init__()
        
        # Load pretrained weights if provided
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}...")
            
            # The checkpoint is saved as: nn.Sequential(*list(models.resnet50().children())[:-2])
            # This means it contains all ResNet50 layers except avgpool and fc
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Check if checkpoint is a state_dict or a model
            if isinstance(checkpoint, nn.Module):
                # Checkpoint is already a model (nn.Sequential)
                self.features = checkpoint
                print("✓ Pretrained encoder loaded successfully (as nn.Module)")
            elif isinstance(checkpoint, dict) and 'state_dict' not in checkpoint:
                # Checkpoint is a state_dict
                print("✓ Checkpoint is a state_dict, loading into ResNet50...")
                resnet = models.resnet50(pretrained=False)
                self.features = nn.Sequential(*list(resnet.children())[:-2])
                
                # Try to load the state dict
                try:
                    self.features.load_state_dict(checkpoint, strict=False)
                    print("✓ Pretrained weights loaded successfully!")
                except Exception as e:
                    print(f"⚠ Warning: Could not load all weights: {e}")
                    print("  Continuing with partial weights...")
            else:
                # Unknown format, create from scratch
                print("⚠ Unknown checkpoint format, creating encoder from scratch...")
                resnet = models.resnet50(pretrained=False)
                self.features = nn.Sequential(*list(resnet.children())[:-2])
            
            print(f"  Encoder architecture: ResNet50 without last 2 layers (no avgpool, no fc)")
            
            # Freeze the pretrained encoder (optional - uncomment to freeze)
            # for param in self.features.parameters():
            #     param.requires_grad = False
        else:
            print(f"Warning: Pretrained checkpoint not found at {pretrained_path}")
            print("Training from scratch...")
            # Create encoder from scratch: ResNet50 without last 2 layers
            resnet = models.resnet50(pretrained=False)
            self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Add adaptive pooling and classifier head
        # Output from features is [batch, 2048, 7, 7] for 224x224 input
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

###################################
#   Training Functions
###################################
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(dataloader):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate the model with detailed metrics"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # For per-class metrics (precision, recall, F1)
    class_correct = [0] * CONFIG['num_classes']
    class_total = [0] * CONFIG['num_classes']
    class_tp = [0] * CONFIG['num_classes']  # True positives
    class_fp = [0] * CONFIG['num_classes']  # False positives
    class_fn = [0] * CONFIG['num_classes']  # False negatives
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class statistics
            for label, pred in zip(labels, predicted):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
                    class_tp[pred] += 1
                else:
                    class_fn[label] += 1  # False negative for true class
                    class_fp[pred] += 1   # False positive for predicted class
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(dataloader):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(CONFIG['classes']):
        metrics = {}
        
        # Accuracy
        if class_total[i] > 0:
            metrics['accuracy'] = 100. * class_correct[i] / class_total[i]
        else:
            metrics['accuracy'] = 0.0
        
        # Precision = TP / (TP + FP)
        if (class_tp[i] + class_fp[i]) > 0:
            metrics['precision'] = 100. * class_tp[i] / (class_tp[i] + class_fp[i])
        else:
            metrics['precision'] = 0.0
        
        # Recall = TP / (TP + FN)
        if (class_tp[i] + class_fn[i]) > 0:
            metrics['recall'] = 100. * class_tp[i] / (class_tp[i] + class_fn[i])
        else:
            metrics['recall'] = 0.0
        
        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        if (metrics['precision'] + metrics['recall']) > 0:
            metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0
        
        metrics['support'] = class_total[i]
        per_class_metrics[class_name] = metrics
    
    return epoch_loss, epoch_acc, per_class_metrics

###################################
#   Main Training Loop
###################################
def train_model():
    """Main training function"""
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(CONFIG['output_dir']) / f'train_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🍃 COFFEE LEAF DISEASE CLASSIFICATION - FINE-TUNING")
    print(f"{'='*70}\n")
    
    # Save configuration
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=4)
    print(f"✓ Configuration saved to {config_path}\n")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = CoffeeLeafDataset(
        CONFIG['data_root'], 
        split='train', 
        transform=train_transform
    )
    val_dataset = CoffeeLeafDataset(
        CONFIG['data_root'], 
        split='valid', 
        transform=val_transform
    )
    
    # Handle class imbalance
    train_labels = train_dataset.get_labels()
    class_weights = calculate_class_weights(
        train_labels, 
        CONFIG['num_classes'], 
        scheme=CONFIG['weight_scheme']
    )
    
    print(f"\n{'='*70}")
    print("Class Weights (for loss function):")
    for i, class_name in enumerate(CONFIG['classes']):
        print(f"  {class_name:12s}: {class_weights[i]:.4f}")
    print(f"{'='*70}\n")
    
    # Create weighted sampler if enabled
    sampler = None
    shuffle = True
    if CONFIG['use_weighted_sampler']:
        print("Using WeightedRandomSampler for balanced training batches...")
        # Calculate sample weights
        sample_weights = torch.tensor([class_weights[label] for label in train_labels])
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False  # Cannot use shuffle with sampler
        print(f"✓ Sampler created with {len(sample_weights)} samples\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    print(f"\n{'='*70}")
    print("Creating model...")
    
    # Create model
    model = CoffeeLeafClassifier(
        num_classes=CONFIG['num_classes'],
        pretrained_path=CONFIG['checkpoint_path']
    ).to(device)
    
    # Loss and optimizer
    if CONFIG['use_class_weights']:
        print(f"Using weighted CrossEntropyLoss with class weights")
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        print(f"Using standard CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✓ Optimizer: AdamW (lr={CONFIG['learning_rate']})")
    print(f"✓ Scheduler: CosineAnnealingLR")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"\n{'='*70}")
    print(f"Starting training for {CONFIG['num_epochs']} epochs...")
    print(f"{'='*70}\n")
    
    # Training loop
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        print(f"{'-'*70}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, per_class_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        print(f"\n  Per-class Validation Metrics:")
        print(f"  {'Class':<12} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Support':>8}")
        print(f"  {'-'*60}")
        for class_name, metrics in per_class_metrics.items():
            print(f"  {class_name:<12} {metrics['accuracy']:>6.1f}% {metrics['precision']:>6.1f}% "
                  f"{metrics['recall']:>6.1f}% {metrics['f1']:>6.1f}% {metrics['support']:>8d}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'per_class_metrics': per_class_metrics,
                'config': CONFIG
            }, best_model_path)
            print(f"\n  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % CONFIG['save_every'] == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': CONFIG
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path.name}")
    
    # Save final model
    final_model_path = checkpoint_dir / 'final_model.pth'
    torch.save({
        'epoch': CONFIG['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'config': CONFIG
    }, final_model_path)
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📊 Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"\n💾 Saved files:")
    print(f"  - Best model: {checkpoint_dir / 'best_model.pth'}")
    print(f"  - Final model: {checkpoint_dir / 'final_model.pth'}")
    print(f"  - Training history: {history_path}")
    print(f"  - Configuration: {config_path}")
    
    # Plot training curves
    plot_training_curves(history, output_dir)
    
    return model, history, output_dir

###################################
#   Visualization
###################################
def plot_training_curves(history, output_dir):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Best metrics summary
    best_val_acc = max(history['val_acc'])
    best_val_epoch = history['val_acc'].index(best_val_acc) + 1
    final_val_acc = history['val_acc'][-1]
    
    summary_text = f"""
    Training Summary:
    
    Best Val Accuracy: {best_val_acc:.2f}%
    Best Epoch: {best_val_epoch}
    
    Final Val Accuracy: {final_val_acc:.2f}%
    Final Train Accuracy: {history['train_acc'][-1]:.2f}%
    
    Total Epochs: {len(epochs)}
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    plot_path = output_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training curves saved to {plot_path}")
    plt.close()

###################################
#   Main Entry Point
###################################
if __name__ == "__main__":
    # Train the model
    model, history, output_dir = train_model()
    
    print(f"\n{'='*70}")
    print("🎉 All done! You can find the results in:")
    print(f"   {output_dir}")
    print(f"{'='*70}\n")
