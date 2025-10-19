#!/usr/bin/env python3
"""
Grad-CAM Visualization Script
Visualizes attention maps of ResNet50 model on test samples
"""

import os
import random
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# Configuration
DATA_ROOT = "data/test2"  # Pre-cropped test images
RESNET_WEIGHTS = "ckpt/final_model2.pth"  # Trained ResNet50
IMG_SIZE = 224
NUM_SAMPLES_PER_CLASS = None  # None = process ALL samples, or set a number (e.g., 3, 5, 10)
OUTPUT_DIR = "gradcam_visualizations"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Class names (same order as training)
CLASS_NAMES = ['Cercospora', 'Corticium', 'mealy', 'Miner', 'Phoma', 'Rust']

# Transform for model input
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Inverse transform for visualization
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


class CoffeeLeafClassifier(nn.Module):
    """ResNet50 based classifier matching train.py architecture"""
    def __init__(self, num_classes=6):
        super(CoffeeLeafClassifier, self).__init__()
        # Encoder: ResNet50 without last 2 layers (no avgpool, no fc)
        resnet = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Add adaptive pooling and classifier head
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


class GradCAM:
    """Grad-CAM implementation for visualization"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """Generate Grad-CAM heatmap"""
        # Forward pass
        model_output = self.model(input_image)
        
        if target_class is None:
            target_class = model_output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_score = model_output[:, target_class]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam, target_class, model_output


def apply_colormap_on_image(org_im, activation_map, colormap_name='jet'):
    """Apply colormap on heatmap and overlay on original image"""
    # Resize activation map to match image size
    height, width = org_im.size[1], org_im.size[0]
    activation_map_resized = Image.fromarray(activation_map).resize((width, height), Image.BILINEAR)
    activation_map_resized = np.array(activation_map_resized)
    
    # Apply colormap
    colormap = plt.get_cmap(colormap_name)
    heatmap = colormap(activation_map_resized)
    heatmap = np.uint8(255 * heatmap[:, :, :3])
    
    # Convert original image to array
    org_im_array = np.array(org_im)
    
    # Overlay
    overlay = np.uint8(0.6 * org_im_array + 0.4 * heatmap)
    
    return overlay


def load_model(checkpoint_path, num_classes=6):
    """Load trained model"""
    model = CoffeeLeafClassifier(num_classes=num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Extract model_state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    
    return model


def collect_samples(data_root, num_samples_per_class=None):
    """Collect samples from each class
    
    Args:
        data_root: Root directory containing class folders
        num_samples_per_class: If None, collect all samples. Otherwise, randomly select this many.
    """
    samples = {}
    data_root = Path(data_root)
    
    for class_name in CLASS_NAMES:
        class_dir = data_root / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist!")
            continue
        
        # Get all image files
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + \
                     list(class_dir.glob("*.png")) + list(class_dir.glob("*.JPG"))
        
        # Select samples
        if num_samples_per_class is None:
            # Use all samples
            selected = image_files
        elif len(image_files) >= num_samples_per_class:
            # Randomly select specified number
            selected = random.sample(image_files, num_samples_per_class)
        else:
            # Use all available if less than requested
            selected = image_files
        
        samples[class_name] = selected
    
    return samples


def visualize_gradcam(model, gradcam, image_path, class_idx, output_path):
    """Visualize single image with Grad-CAM"""
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad = True
    
    # Generate Grad-CAM
    cam, predicted_class, model_output = gradcam.generate_cam(input_tensor, target_class=class_idx)
    
    # Get prediction probabilities
    probs = F.softmax(model_output, dim=1).squeeze().cpu().detach().numpy()
    
    # Resize CAM to 0-255 range for visualization
    cam_uint8 = np.uint8(255 * cam)
    
    # Apply colormap
    overlay = apply_colormap_on_image(original_image, cam_uint8, 'jet')
    
    # Create visualization
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Original image
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(original_image)
    ax1.set_title(f'Original Image\nTrue: {CLASS_NAMES[class_idx]}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Heatmap
    ax2 = fig.add_subplot(gs[:, 1])
    im = ax2.imshow(cam, cmap='jet')
    ax2.set_title(f'Grad-CAM Heatmap\nPred: {CLASS_NAMES[predicted_class]}', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Overlay
    ax3 = fig.add_subplot(gs[:, 2])
    ax3.imshow(overlay)
    ax3.set_title(f'Overlay\nConf: {probs[predicted_class]:.2%}', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Add prediction probabilities as bar chart (removed for cleaner layout)
    # Instead, add text summary
    fig.text(0.5, 0.02, f'File: {Path(image_path).name}', 
             ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return predicted_class, probs


def create_summary_grid(model, gradcam, samples, output_dir):
    """Create a summary grid showing all samples"""
    output_dir = Path(output_dir)
    
    for class_name, image_paths in samples.items():
        if len(image_paths) == 0:
            continue
        
        class_idx = CLASS_NAMES.index(class_name)
        
        # Create figure for this class
        n_samples = len(image_paths)
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Grad-CAM Visualizations: {class_name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        for idx, image_path in enumerate(image_paths):
            # Load image
            original_image = Image.open(image_path).convert('RGB')
            input_tensor = transform(original_image).unsqueeze(0).to(DEVICE)
            input_tensor.requires_grad = True
            
            # Generate Grad-CAM
            cam, predicted_class, model_output = gradcam.generate_cam(input_tensor, target_class=class_idx)
            probs = F.softmax(model_output, dim=1).squeeze().cpu().detach().numpy()
            
            # Resize and apply colormap
            cam_uint8 = np.uint8(255 * cam)
            overlay = apply_colormap_on_image(original_image, cam_uint8, 'jet')
            
            # Plot
            axes[idx, 0].imshow(original_image)
            axes[idx, 0].set_title(f'Original\n{Path(image_path).name[:20]}...', fontsize=10)
            axes[idx, 0].axis('off')
            
            im = axes[idx, 1].imshow(cam, cmap='jet')
            axes[idx, 1].set_title(f'Heatmap\nPred: {CLASS_NAMES[predicted_class]}', fontsize=10)
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(overlay)
            axes[idx, 2].set_title(f'Overlay\nConf: {probs[predicted_class]:.2%}', fontsize=10)
            axes[idx, 2].axis('off')
        
        plt.tight_layout()
        grid_path = output_dir / f'{class_name}_gradcam_grid.png'
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved grid: {grid_path}")


def main():
    """Main function"""
    print(f"{'='*60}")
    print(f"Grad-CAM Visualization for Coffee Leaf Disease Classification")
    print(f"{'='*60}\n")
    
    print(f"Device: {DEVICE}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Checkpoint: {RESNET_WEIGHTS}")
    
    # Check if we should process all samples or just a subset
    if NUM_SAMPLES_PER_CLASS is None or NUM_SAMPLES_PER_CLASS == 0:
        print(f"Mode: Processing ALL samples in test set\n")
        use_all_samples = True
    else:
        print(f"Mode: Processing {NUM_SAMPLES_PER_CLASS} samples per class\n")
        use_all_samples = False
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(RESNET_WEIGHTS, num_classes=len(CLASS_NAMES))
    print("✓ Model loaded successfully\n")
    
    # Initialize Grad-CAM
    # Target the last convolutional layer in the features
    target_layer = model.features[-1]  # Last layer of ResNet50 features
    gradcam = GradCAM(model, target_layer)
    print(f"✓ Grad-CAM initialized on layer: {target_layer.__class__.__name__}\n")
    
    # Collect samples
    print("Collecting samples...")
    if use_all_samples:
        samples = collect_samples(DATA_ROOT, num_samples_per_class=None)
    else:
        samples = collect_samples(DATA_ROOT, num_samples_per_class=NUM_SAMPLES_PER_CLASS)
    
    total_samples = sum(len(paths) for paths in samples.values())
    print(f"✓ Collected {total_samples} samples from {len(samples)} classes")
    for class_name, paths in samples.items():
        print(f"  - {class_name}: {len(paths)} images")
    print()
    
    # Visualize individual samples
    print("Generating individual visualizations...")
    results = []
    processed = 0
    
    for class_name, image_paths in samples.items():
        class_idx = CLASS_NAMES.index(class_name)
        print(f"\nProcessing {class_name} ({len(image_paths)} images)...")
        
        for idx, img_path in enumerate(image_paths, 1):
            output_path = output_dir / f"{class_name}_{Path(img_path).stem}_gradcam.png"
            pred_class, probs = visualize_gradcam(model, gradcam, img_path, class_idx, output_path)
            
            results.append({
                'true_class': class_name,
                'pred_class': CLASS_NAMES[pred_class],
                'confidence': probs[pred_class],
                'correct': class_idx == pred_class,
                'file': Path(img_path).name
            })
            
            processed += 1
            if idx % 10 == 0 or idx == len(image_paths):
                print(f"  Progress: {idx}/{len(image_paths)} ({processed}/{total_samples} total)")
    
    # Create summary grids (only if processing small number of samples)
    if not use_all_samples or total_samples <= 30:
        print("\nGenerating summary grids...")
        create_summary_grid(model, gradcam, samples, output_dir)
    else:
        print("\nSkipping summary grids (too many samples - would be too large)")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for class_name in CLASS_NAMES:
        class_results = [r for r in results if r['true_class'] == class_name]
        if class_results:
            class_correct = sum(1 for r in class_results if r['correct'])
            class_total = len(class_results)
            print(f"  {class_name:12s}: {class_correct}/{class_total} ({class_correct/class_total*100:.2f}%)")
    
    # Save results to JSON
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"gradcam_results_{timestamp}.json"
    
    summary = {
        'timestamp': timestamp,
        'data_root': str(DATA_ROOT),
        'checkpoint': str(RESNET_WEIGHTS),
        'total_samples': total,
        'correct_predictions': correct,
        'accuracy': correct / total,
        'per_class_accuracy': {
            class_name: {
                'correct': sum(1 for r in results if r['true_class'] == class_name and r['correct']),
                'total': len([r for r in results if r['true_class'] == class_name]),
                'accuracy': sum(1 for r in results if r['true_class'] == class_name and r['correct']) / 
                           len([r for r in results if r['true_class'] == class_name])
            }
            for class_name in CLASS_NAMES
        },
        'detailed_results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    print(f"\nAll visualizations saved to: {output_dir}/")
    print(f"{'='*60}\n")
    
    print("✅ Grad-CAM visualization completed!")


if __name__ == "__main__":
    main()
