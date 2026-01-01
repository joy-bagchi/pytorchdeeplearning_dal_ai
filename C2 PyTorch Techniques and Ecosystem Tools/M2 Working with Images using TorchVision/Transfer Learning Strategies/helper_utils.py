import json
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import requests
import torch
import torchmetrics
import torchvision
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm



def unnormalize(tensor):
    """
    Reverses the normalization of a PyTorch image tensor.

    This function takes a normalized tensor and applies the inverse
    transformation to return the pixel values to the standard [0, 1] range.
    The mean and standard deviation values used for the original
    normalization are hardcoded within this function.

    Args:
        tensor (torch.Tensor): The normalized input tensor with a shape of
                               (C, H, W), where C is the number of channels.

    Returns:
        torch.Tensor: The unnormalized tensor with pixel values clamped to
                      the valid [0, 1] range.
    """
    # Define the mean and standard deviation used for the original normalization.
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    # Create a copy of the tensor to avoid modifying the original in-place.
    unnormalized_tensor = tensor.clone()
    
    # Apply the unnormalization formula to each channel: (pixel * std) + mean.
    for i, (m, s) in enumerate(zip(mean, std)):
        unnormalized_tensor[i].mul_(s).add_(m)
        
    # Clamp pixel values to the valid [0, 1] range to correct for floating-point inaccuracies.
    unnormalized_tensor = torch.clamp(unnormalized_tensor, 0, 1)
    
    # Return the unnormalized tensor.
    return unnormalized_tensor



def create_emnist_dataloaders(batch_size, transform):
    """
    Downloads the EMNIST 'digits' dataset and creates training and validation data loaders
    by splitting the original training set.

    This function downloads the EMNIST 'digits' training data if it is not found locally.
    It then splits this single dataset into an 80% training set and a 20% validation set.
    Finally, it wraps both new datasets in PyTorch DataLoader objects.

    Args:
        batch_size (int): The number of images per batch.
        transform (callable): A function or transform to be applied to the dataset images.

    Returns:
        tuple: A tuple containing the new training and validation data loaders.
    """
    # Define the path for the EMNIST data.
    emnist_data_path = './EMNIST_data'
    
    # Check if the data needs to be downloaded.
    emnist_download = not os.path.exists(emnist_data_path)
    
    # Initialize the full EMNIST digits dataset from the Test split.
    full_train_dataset = torchvision.datasets.EMNIST(
        root=emnist_data_path,
        split='digits',
        train=False,
        download=emnist_download,
        transform=transform
    )

    # Calculate the sizes for the 80/20 split.
    dataset_size = len(full_train_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    # Split the dataset into training and validation subsets.
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])

    # Manually attach the 'classes' attribute to the Subset objects, as random_split
    # does not transfer this attribute automatically.
    train_subset.classes = full_train_dataset.classes
    val_subset.classes = full_train_dataset.classes

    # Create a DataLoader for the training subset with shuffling enabled.
    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=batch_size,
        shuffle=True
    )

    # Create a DataLoader for the validation subset with shuffling disabled.
    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader



def load_imagenet_classes(json_path):
    """
    Loads the ImageNet class index from a JSON file.

    This function checks if the class index file exists locally. If it does not,
    it attempts to download the file from a predefined URL and saves it to the
    specified path before loading its contents.

    Args:
        json_path (str): The local file path for the ImageNet class index JSON file.

    Returns:
        dict: A dictionary containing the ImageNet class index, or None if an
              error occurs during loading or downloading.
    """
    # Initialize the variable to hold the class index data.
    imagenet_classes = None

    # Check if the class index file already exists at the specified path.
    if os.path.exists(json_path):
        print(f"Loading ImageNet class index from: {json_path}")
        try:
            # Open and load the JSON file.
            with open(json_path, 'r') as f:
                imagenet_classes = json.load(f)
            print("Successfully loaded the class index.")
        # Handle cases where the file is not valid JSON.
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {json_path}")
            return None
        # Handle other potential exceptions during file reading.
        except Exception as e:
            print(f"Error reading {json_path}: {e}")
            return None
    else:
        # If the file does not exist, download it.
        print("Downloading ImageNet class index...")
        # URL for the ImageNet class index file.
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        try:
            # Send an HTTP GET request to the URL.
            response = requests.get(url)
            # Raise an exception for bad status codes (4xx or 5xx).
            response.raise_for_status()
            # Write the downloaded content to the specified local file path.
            with open(json_path, 'wb') as f:
                f.write(response.content)
            # Open and load the newly created JSON file.
            with open(json_path, 'r') as f:
                imagenet_classes = json.load(f)
            print("Download complete and class index loaded.\n")
        # Handle network-related errors during download.
        except requests.exceptions.RequestException as e:
            print(f"Error downloading class index: {e}")
            return None
        # Handle cases where the downloaded content is not valid JSON.
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format downloaded from {url}")
            return None
        # Handle any other unexpected errors.
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
            
    # Return the loaded class index dictionary.
    return imagenet_classes



def show_predictions(model, val_loader, device, class_names):
    """
    Displays a grid of predictions for a sample of images.

    This function efficiently selects one image for each of up to nine
    distinct classes from the data loader. It then runs a single batch
    inference and visualizes the images along with their actual and
    predicted labels.

    Args:
        model (torch.nn.Module): The trained model to use for predictions.
        val_loader (torch.utils.data.DataLoader): The data loader containing
            the validation dataset.
        device (torch.device): The device (e.g., 'cpu' or 'cuda') on which
            to run the model.
        class_names (dict or list): A list or dictionary that maps class
            indices to their string names.
    """
    # Set the model to evaluation mode
    model.eval()
    # Move the model to the specified device
    model.to(device)

    # Dictionary to store one image tensor per class label
    images_to_show = {}
    # Iterate through the data loader to find unique class images
    for images, labels in val_loader:
        # Check each image in the current batch
        for i in range(len(labels)):
            # Get the integer label for the current image
            label = labels[i].item()
            # If this class has not been seen yet, store the image
            if label not in images_to_show:
                # Store the image on the CPU to conserve GPU memory
                images_to_show[label] = images[i].cpu()
            # Stop searching once 9 unique images are collected
            if len(images_to_show) >= 9:
                break
        # Exit the outer loop as well if 9 images have been found
        if len(images_to_show) >= 9:
            break
    
    # Determine the number of images to display
    num_to_display = len(images_to_show)
    # Handle the case where the validation loader is empty
    if num_to_display == 0:
        print("No images found in the validation loader.")
        return
        
    # Prepare the collected images for batched inference
    # Get the labels of the images that will be displayed
    actual_labels = list(images_to_show.keys())
    # Create a batch tensor from the collected images and move it to the device
    image_batch = torch.stack(list(images_to_show.values())).to(device)

    # Disable gradient calculations for inference
    with torch.no_grad():
        # Get model predictions for the image batch
        outputs = model(image_batch)
        # Find the index of the highest prediction score for each image
        _, predicted_indices = torch.max(outputs, 1)
    
    # Move predictions to CPU and convert to a NumPy array
    predicted_indices = predicted_indices.cpu().numpy()

    # Create a subplot to display the images and predictions
    # Initialize a 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    # Set a title for the entire figure
    fig.suptitle('Model Predictions', fontsize=16)

    # Iterate through the collected images to display them
    for i, label in enumerate(actual_labels):
        # Select the appropriate subplot for the current image
        ax = axes[i // 3, i % 3]
        
        # Retrieve the image tensor
        image = images_to_show[label]
        # Get the predicted class index for this image
        predicted_class_idx = predicted_indices[i]

        # Determine the actual and predicted class names
        # Check if class_names is a dictionary (e.g., ImageNet format)
        if isinstance(class_names, dict):
            # Handles ImageNet's dictionary format
            predicted_class_name = class_names[str(predicted_class_idx)][1]
            actual_class = str(label)
            # Set title color
            title_color = 'red'
        # Assumes class_names is a list for custom datasets
        else:
            # Handles custom datasets with a list of class names
            predicted_class_name = class_names[predicted_class_idx]
            actual_class = class_names[label]
            # Set title color to green for correct predictions and red for incorrect
            title_color = 'green' if actual_class == predicted_class_name else 'red'
            
        # Un-normalize the image tensor for visualization
        unnormalized_image = unnormalize(image).permute(1, 2, 0).numpy()
        # Display the image
        ax.imshow(unnormalized_image)
        # Set the title with actual and predicted labels, colored by correctness
        ax.set_title(f"Actual: {actual_class}\nPredicted: {predicted_class_name}", color=title_color)
        # Hide the axes for a cleaner look
        ax.axis('off')

    # Hide any unused subplots if less than 9 images are shown
    for j in range(num_to_display, 9):
        axes[j // 3, j % 3].axis('off')

    # Adjust subplot parameters for a tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Display the plot
    plt.show()
    
    
    
def training_loop(model, trainloader, valloader, loss_function, optimizer, num_epochs, device):
    """
    Executes the training and validation loop for a given model.

    This function iterates over the training dataset for a specified number of epochs,
    updating the model's weights. After each epoch, it evaluates the model's
    performance on the validation dataset and calculates the final accuracy.

    Args:
        model: The neural network model to be trained.
        trainloader: The DataLoader for the training dataset.
        valloader: The DataLoader for the validation dataset.
        loss_function: The criterion to compute the loss.
        optimizer: The optimization algorithm to update model weights.
        num_epochs (int): The total number of training epochs.
        device: The computing device ('cuda' or 'cpu') to run the training on.

    Returns:
        The trained model after completing all epochs.
    """
    # Determine the number of classes from the training dataset's properties.
    num_classes = len(trainloader.dataset.classes)

    # Transfer the model to the specified computation device.
    model.to(device)

    # Initialize the accuracy metric from torchmetrics for multiclass classification.
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)

    # Begin the main loop for training over the number of epochs.
    for epoch in range(num_epochs):
        # Set the model to training mode.
        model.train()
        # Initialize running loss for the current epoch.
        running_loss = 0.0
        # Create a tqdm progress bar for the training loader.
        train_tqdm = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        # Iterate over batches of data from the training loader.
        for inputs, labels in train_tqdm:
            # Move inputs and labels to the specified device.
            inputs, labels = inputs.to(device), labels.to(device)
            # Clear the gradients of all optimized tensors.
            optimizer.zero_grad()
            # Perform a forward pass to get model outputs.
            outputs = model(inputs)
            # Calculate the loss.
            loss = loss_function(outputs, labels)
            # Perform backpropagation to compute gradients.
            loss.backward()
            # Update the model's parameters.
            optimizer.step()
            # Accumulate the loss for the batch.
            running_loss += loss.item()
            # Update the postfix of the progress bar to show the current average loss.
            train_tqdm.set_postfix({'loss': running_loss / (train_tqdm.n + 1)})

        # Set the model to evaluation mode for validation.
        model.eval()
        # Create a tqdm progress bar for the validation loader.
        val_tqdm = tqdm(valloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")
        # Disable gradient calculations for the validation phase.
        with torch.no_grad():
            # Iterate over batches of data from the validation loader.
            for images, labels in val_tqdm:
                # Move images and labels to the specified device.
                images, labels = images.to(device), labels.to(device)
                # Get the model's predictions for the validation images.
                outputs = model(images)
                # Determine the predicted class by finding the index of the maximum logit.
                predicted = torch.argmax(outputs, 1)
                # Update the accuracy metric with the predictions and true labels.
                accuracy_metric.update(predicted, labels)

    # Print a message indicating that training is complete.
    print("\nFinished Training!")

    # Compute the final accuracy over the entire validation dataset.
    final_accuracy = accuracy_metric.compute()
    # Print the final validation accuracy.
    print(f"Final Validation Accuracy: {final_accuracy:.4f}")
    
    # Return the trained model.
    return model