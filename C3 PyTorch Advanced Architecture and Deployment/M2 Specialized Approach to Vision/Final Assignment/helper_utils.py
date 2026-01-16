import torch
from torchvision.models import resnet50
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import make_grid
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from PIL import Image
import os

def display_cam(image_tensor, cam_up):
    # De-normalise for viewing (assumes ImageNet stats)
    means = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    stds = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb = (image_tensor[0].cpu() * stds + means).clamp(0, 1).permute(1, 2, 0).numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.imshow(cam_up.cpu().detach().cpu(), cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.title("Simplified CAM Overlay")
    plt.show()
    

def preprocess_image(image_path, device="cpu"):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    image_tensor = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    return image_tensor

def print_model_architecture(net):
    # Print the model architecture
    print("\nLayer-wise Details:")
    print("-" * 80)
    for name, module in net.named_children():
        params = sum(p.numel() for p in module.parameters())
        layer_type = ""
        if isinstance(module, torch.nn.Conv2d):
            layer_type = f"(Conv {module.kernel_size[0]}x{module.kernel_size[1]})"
        print(f"Layer: {name:<20} {layer_type:<15} Parameters: {params:,}")
        
        # For sequential blocks, show sub-layers
        if isinstance(module, torch.nn.Sequential):
            for sub_name, sub_module in module.named_children():
                sub_params = sum(p.numel() for p in sub_module.parameters())
                sub_type = ""
                if isinstance(sub_module, torch.nn.Conv2d):
                    sub_type = f"(Conv {sub_module.kernel_size[0]}x{sub_module.kernel_size[1]})"
                print(f"  └─ {sub_name:<18} {sub_type:<15} Parameters: {sub_params:,}")
    print("-" * 80)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total Parameters: {total_params:,}")

def load_model(model_path, device="cpu"):
    print("Starting loading model...")
    net = resnet50(weights=None)
    print("Changing the final layer to a binary classification layer...")
    net.fc = torch.nn.Linear(net.fc.in_features, 2)
    print("Loading the model weights...")
    loaded_sd = torch.load(model_path, map_location=device)
    print("Loading the model weights into the model...")
    net.load_state_dict(loaded_sd, strict=False)
    print("Model loaded successfully!")
    return net

def print_samples_from_dataset(dataset_path, k):     
    # Print k sample filenames for each class
    print(f"\nShowing {k} sample filenames per class:")
    print("-" * 50)

    for class_name in sorted(os.listdir(dataset_path)):
        class_dir = os.path.join(dataset_path, class_name)
        
        if os.path.isdir(class_dir):
            print(f"\n{class_dir}/")
            # Get all valid image files
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith('.jpg')]
            # Print first k files
            for filename in sorted(image_files[:k]):
                print(f"... {filename}")

def plot_samples_from_dataset(dataset_path):
    # Create figure for image grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    # Import required modules
    import os
    from collections import defaultdict

    # Counter for plotting images
    plot_idx = 0

    # Process each class directory
    for class_name in sorted(os.listdir(dataset_path)):
        class_dir = os.path.join(dataset_path, class_name)
        
        if os.path.isdir(class_dir):
            # Get first image from the class directory
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith('.jpg')]
            if image_files:
                # Load and display first image
                img_path = os.path.join(class_dir, image_files[0])
                img = Image.open(img_path).convert("RGB")
                # img.show()
                axes[plot_idx].imshow(img)
                axes[plot_idx].set_title(f"{class_name}")
                axes[plot_idx].axis('off')
                plot_idx += 1

    plt.tight_layout()

def grid_from_layer(feat: torch.Tensor, num_filters: int = 4) -> np.ndarray:
    """
    Select `num_filters` channels at equal intervals and tile them.
    """
    feat = feat.squeeze().cpu().numpy()  # C,H,W
    C, H, W = feat.shape
    idxs = np.linspace(0, C - 1, num_filters, dtype=int)

    grid_size = int(np.ceil(np.sqrt(num_filters)))
    grid = np.zeros((grid_size * H, grid_size * W))

    for i, c in enumerate(idxs):
        r, col = divmod(i, grid_size)
        grid[
            r * H : (r + 1) * H, col * W : (col + 1) * W
        ] = feat[c]

    # normalise 0-1 for imshow
    grid -= grid.min()
    grid /= grid.max() + 1e-5
    return grid


def visual_strip(upsampled: List[torch.Tensor]):
    grid = make_grid(torch.cat(upsampled), nrow=5, padding=2)
    plt.figure(figsize=(14, 3))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title("Feature-map progression conv1 → layer4")
    plt.show()

def display_saliency(image_tensor: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.5):
        # Denormalise for display if ImageNet stats were used
        # (adjust these values if you use a different preprocessing pipeline)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_disp = image_tensor.detach()[0].cpu() * std + mean
        img_disp = img_disp.clamp(0, 1).permute(1, 2, 0).numpy()

        plt.figure(figsize=(6, 6))
        plt.imshow(img_disp)
        plt.imshow(heatmap.cpu(), cmap="jet", alpha=alpha)
        plt.axis("off")
        plt.title("Salience Map Overlay")
        plt.show()

def display_feature_hierarchy(activations, image_path):
    # Preprocess the image for display
    preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
   # Create plot
    plt.figure(figsize=(15, 10))

    # Original image (de-normalised for display)
    orig = img[0].cpu()
    orig = orig * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    orig = orig + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    plt.subplot(2, 3, 1)
    plt.imshow(orig.permute(1, 2, 0).clamp(0, 1))
    plt.title("Original Image")
    plt.axis("off")

    # Feature-map grids
    order = ["conv1", "layer1", "layer2", "layer3", "layer4"]
    for sp, name in enumerate(order, start=2):
        grid = grid_from_layer(activations[name])
        plt.subplot(2, 3, sp)
        plt.imshow(grid, cmap="viridis")
        plt.title(f"{name}: {activations[name].shape[1]} filters")
        plt.axis("off")

    plt.tight_layout()
    
    

    plt.show()

    