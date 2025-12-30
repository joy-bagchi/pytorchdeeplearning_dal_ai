import os
import random

import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from tqdm.auto import tqdm



AIvsReal_path = "./AIvsReal_sampled"



def show_random_images(split, category, dataset_path, num_images=5, seed=6):
    """Displays a specified number of random images from a given dataset category.

    Args:
        split: The dataset split to choose images from (e.g., 'train', 'test').
        category: The category or class of images to display.
        dataset_path: The root path to the dataset directory.
        num_images: The number of random images to display.
        seed: The random seed for reproducibility of image selection.
    """
    # Construct the path to the target folder
    folder = os.path.join(dataset_path, split, category)
    # Set the random seed for reproducibility
    random.seed(seed)
    # Randomly sample a list of image filenames from the folder
    image_files = random.sample(os.listdir(folder), num_images)

    # Create a figure to display the images
    plt.figure(figsize=(15, 3))
    # Loop through the selected image files and display them
    for i, fname in enumerate(image_files):
        # Get the full path of the image
        img_path = os.path.join(folder, fname)
        # Open and convert the image to RGB
        img = Image.open(img_path).convert("RGB")
        # Create a subplot for the current image
        plt.subplot(1, num_images, i + 1)
        # Display the image
        plt.imshow(img)
        # Assign the category to a variable for the title
        group = category
        # Set the title for the subplot
        plt.title(f"{split}/{group}")
        # Turn off the axes for a cleaner look
        plt.axis("off")
    # Show the final plot with all images
    plt.show()


    
def get_data_loaders(transform, batch_size, AIvsReal_path):
    """Creates and returns training and validation data loaders for a dataset.

    Args:
        transform: The torchvision transforms to be applied to the images.
        batch_size: The number of samples per batch in the data loaders.
        AIvsReal_path: The root path to the dataset directory.

    Returns:
        A tuple containing the training and validation data loaders.
    """
    # Load the full dataset from the specified training path
    dataset = datasets.ImageFolder(root=AIvsReal_path + "/train/", transform=transform)

    # Determine the size of the training set (80% of the dataset)
    train_size = int(0.8 * len(dataset))
    # Determine the size of the validation set (the remaining 20%)
    val_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create the DataLoader for the training set
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # Create the DataLoader for the validation set
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Return the created data loaders
    return train_loader, val_loader



def training_epoch(model, train_loader, optimizer, loss_function, device, epoch, num_epochs, emty_cache=True, silent=False,):
    """Performs a single training epoch for a given model.

    Args:
        model: The neural network model to be trained.
        train_loader: The DataLoader for the training data.
        optimizer: The optimization algorithm to update model weights.
        loss_function: The loss function used to evaluate model performance.
        device: The device (e.g., 'cuda' or 'cpu') to run the training on.
        epoch: The current epoch number.
        num_epochs: The total number of epochs for training.
        emty_cache: A boolean flag to empty the CUDA cache after each batch.
        silent: A boolean flag to disable the progress bar and print statements.
    
    Returns:
        The average loss for the epoch.
    """
    # Set the model to training mode
    model.train()

    # Initialize the total loss for the epoch
    running_loss = 0.0
    # Create a TQDM progress bar for the training loader
    train_loader_tqdm = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{num_epochs} - Training",
        leave=False,
        disable=silent,
    )

    # Iterate over the batches of data in the training loader
    for i, (images, labels) in enumerate(train_loader_tqdm):
        # Move images and labels to the specified device
        images = images.to(device)
        labels = labels.to(device)

        # Perform a forward pass to get the model's predictions
        outputs = model(images)
        # Calculate the loss between the predictions and true labels
        loss = loss_function(outputs, labels)

        # Clear the gradients from the previous iteration
        optimizer.zero_grad()
        # Perform backpropagation to compute gradients
        loss.backward()
        # Update the model's weights using the optimizer
        optimizer.step()

        # Add the current batch's loss to the running total
        running_loss += loss.item()

        # Update the progress bar with the current loss if not in silent mode
        if not silent:
            train_loader_tqdm.set_postfix(loss=loss.item(), refresh=False)

        # Print training progress at specified intervals if not in silent mode
        if ((i + 1) % 45 == 0) & (not silent):
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )

        # Optionally, clear memory to prevent out-of-memory errors
        if emty_cache:
            del images, labels, outputs, loss
            torch.cuda.empty_cache()

    # Calculate the average loss for the entire epoch
    epoch_loss = running_loss / len(train_loader)
    # Return the calculated average epoch loss
    return epoch_loss



def evaluate_model(model, val_loader, device, silent=False):
    """
    Evaluates the performance of a model on a validation dataset.

    Args:
        model: The model to be evaluated.
        val_loader: DataLoader for the validation dataset.
        device: The device (e.g., 'cpu' or 'cuda') to run the evaluation on.
        silent: If True, suppresses the printing of the validation accuracy.

    Returns:
        The accuracy of the model on the validation dataset.
    """
    # Set the model to evaluation mode
    model.eval()
    # Initialize variables for tracking correct predictions and total samples
    correct, total = 0, 0
    # Disable gradient calculations for inference
    with torch.no_grad():
        # Iterate over the validation data
        for inputs, labels in val_loader:
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            # Perform a forward pass
            outputs = model(inputs)
            # Get the predicted class with the highest probability
            _, predicted = torch.max(outputs, 1)
            # Update the total number of samples
            total += labels.size(0)
            # Update the count of correctly predicted samples
            correct += (predicted == labels).sum().item()

            # Release memory to prevent memory leaks
            del inputs, labels, outputs
            # Clear the GPU cache
            torch.cuda.empty_cache()

    # Calculate the accuracy
    accuracy = correct / total
    # Check if printing the accuracy is enabled
    if not silent:
        # Print the validation accuracy
        print(f"Validation Accuracy: {100 * accuracy:.2f}%")

    # Return the calculated accuracy
    return accuracy



def extract_attr(trial, transform, model, params):
    """
    Sets user-defined attributes on an Optuna trial.

    Args:
        trial: The Optuna trial object.
        transform: The data transformation to be stored.
        model: The machine learning model to be stored.
        params: The parameters of the model or experiment.
    """
    # Set a user attribute for the data transformation
    trial.set_user_attr("transform", transform)
    # Set a user attribute for the model
    trial.set_user_attr("model", model)
    # Set a user attribute for the parameters
    trial.set_user_attr("params_code", params)
