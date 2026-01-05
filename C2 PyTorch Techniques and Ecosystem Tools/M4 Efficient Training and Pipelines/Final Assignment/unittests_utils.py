import os
import shutil
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models as tv_models


def setup_dummy_data_dir(tmp_path="tmp_data"):
    os.makedirs(os.path.join(tmp_path, "train", "NORMAL"), exist_ok=True)
    os.makedirs(os.path.join(tmp_path, "train", "PNEUMONIA"), exist_ok=True)
    os.makedirs(os.path.join(tmp_path, "val", "NORMAL"), exist_ok=True)
    os.makedirs(os.path.join(tmp_path, "val", "PNEUMONIA"), exist_ok=True)

    # Create dummy images
    Image.new('RGB', (10, 10)).save(os.path.join(tmp_path, "train/NORMAL/img1.jpg"))
    Image.new('RGB', (10, 10)).save(os.path.join(tmp_path, "train/NORMAL/img2.jpg"))
    Image.new('RGB', (10, 10)).save(os.path.join(tmp_path, "train/PNEUMONIA/img3.jpg"))
    Image.new('RGB', (10, 10)).save(os.path.join(tmp_path, "val/NORMAL/img4.jpg"))
    Image.new('RGB', (10, 10)).save(os.path.join(tmp_path, "val/PNEUMONIA/img5.jpg"))
    return tmp_path

def cleanup_dummy_data_dir(tmp_path="tmp_data"):
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)


def setup_dummy_weights(path="dummy_weights.pth", num_classes=3):
    """Creates and saves a dummy state_dict for a ResNet model."""
    model = tv_models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    torch.save(model.state_dict(), path)
    return path

def cleanup_dummy_weights(path="dummy_weights.pth"):
    """Removes the dummy weights file."""
    if os.path.exists(path):
        os.remove(path)