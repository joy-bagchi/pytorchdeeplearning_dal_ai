import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import re

def check_color_jitter(learner_transform, expected_brightness, expected_contrast):
    
    color_jitter_found = False
    found_correct_jitter = False
    found_brightness_val = None
    found_contrast_val = None

    for transform in learner_transform.transforms:
        if isinstance(transform, transforms.ColorJitter):
            color_jitter_found = True
            found_brightness = transform.brightness
            found_contrast = transform.contrast

            # Try to infer the input-style value from the range for brightness
            if isinstance(found_brightness, tuple):
                lower = found_brightness[0]
                upper = found_brightness[1]
                if lower >= 0 and abs(upper - 1) == abs(lower - 1):
                    found_brightness_val = abs(lower - 1)
                else:
                    found_brightness_val = f"({lower:.1f}, {upper:.1f})"
            else:
                found_brightness_val = found_brightness

            # Try to infer the input-style value from the range for contrast
            if isinstance(found_contrast, tuple):
                lower = found_contrast[0]
                upper = found_contrast[1]
                if lower >= 0 and abs(upper - 1) == abs(lower - 1):
                    found_contrast_val = abs(lower - 1)
                else:
                    found_contrast_val = f"({lower:.1f}, {upper:.1f})"
            else:
                found_contrast_val = found_contrast

            if found_brightness == (1 - expected_brightness, 1 + expected_brightness) and \
               found_contrast == (1 - expected_contrast, 1 + expected_contrast):
                found_correct_jitter = True
            break

    return color_jitter_found, found_correct_jitter, found_brightness_val, found_contrast_val


def check_shuffle(data_loader, should_shuffle):
    """
    Checks if a DataLoader is shuffling data by observing the order of labels.
    """
    first_iteration_labels = []
    for _, labels in data_loader:
        first_iteration_labels.extend(labels.tolist())

    second_iteration_labels = []
    for _, labels in data_loader:
        second_iteration_labels.extend(labels.tolist())

    if should_shuffle:
        return first_iteration_labels != second_iteration_labels
    else:
        return first_iteration_labels == second_iteration_labels
    
    
def remove_comments(code):
    # This regex pattern matches comments in the code
    pattern = r'#.*'
    
    # Use re.sub() to replace comments with an empty string
    code_without_comments = re.sub(pattern, '', code)
    
    # Split the code into lines, strip each line, and filter out empty lines
    lines = code_without_comments.splitlines()
    non_empty_lines = [line.rstrip() for line in lines if line.strip()]
    
    # Join the non-empty lines back into a single string
    return '\n'.join(non_empty_lines)
    
    
class MockImageFolder(Dataset):
    """
    A mock dataset that behaves like ImageFolder for testing purposes.
    """
    def __init__(self, num_samples, img_size=(32, 32)):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image_array = np.full((self.img_size[0], self.img_size[1], 3), idx % 256, dtype=np.uint8)
        image = Image.fromarray(image_array)
        label = torch.tensor(idx % 2)

        if self.transform:
            image = self.transform(image)
        
        return image, label

def generate_mock_datasets(train_size=100, val_size=75):
    # Create instances of the module-level class
    mock_train_dataset = MockImageFolder(num_samples=train_size)
    mock_val_dataset = MockImageFolder(num_samples=val_size)
    return mock_train_dataset, mock_val_dataset