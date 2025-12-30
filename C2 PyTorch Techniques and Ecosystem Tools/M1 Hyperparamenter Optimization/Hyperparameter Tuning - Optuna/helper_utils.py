import math
from typing import Optional

import torch
import torchmetrics
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import matplotlib as mpl



# Global plot style
PLOT_STYLE = {
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "font.family": "sans",  # "sans-serif",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 3,
    "lines.markersize": 6,
}

mpl.rcParams.update(PLOT_STYLE)

# Custom colors (reusable)
BLUE_COLOR_TRAIN = "#237B94"  # Blue
PINK_COLOR_TEST = "#F65B66"  # Pink



def set_seed(seed=42):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed: The integer value to use as the random seed.
    """
    # Set the seed for PyTorch CPU operations
    torch.manual_seed(seed)
    # Set the seed for PyTorch CUDA operations on all GPUs
    torch.cuda.manual_seed_all(seed)
    # Set the seed for NumPy's random number generator
    np.random.seed(seed)
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Configure CuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    # Disable the CuDNN benchmark mode, which can be non-deterministic
    torch.backends.cudnn.benchmark = False

    

class NestedProgressBar:
    """
    Manages nested tqdm progress bars for loops like epochs and batches.

    This class provides a convenient way to display and control separate
    progress bars for outer and inner loops (e.g., training epochs and
    data batches). It supports both terminal and notebook environments
    and allows for conditional message logging at specified intervals.
    """
    def __init__(
        self,
        total_epochs,
        total_batches,
        g_epochs=None,
        g_batches=None,
        epoch_message_freq=None,
        batch_message_freq=None,
        mode="train",
    ):
        """
        Initializes the nested progress bars.

        Args:
            total_epochs: The total number of epochs for the process.
            total_batches: The total number of batches in one epoch.
            g_epochs: The visual granularity for the epoch bar. If None,
                      it defaults to total_epochs.
            g_batches: The visual granularity for the batch bar. If None,
                       it defaults to total_batches.
            epoch_message_freq: The frequency (in epochs) to log messages.
            batch_message_freq: The frequency (in batches) to log messages.
            mode: The operational mode, either 'train' or 'eval'. 'train'
                  mode shows both epoch and batch bars, while 'eval'
                  mode only shows the batch bar.
        """
        # Set the operational mode ('train' or 'eval')
        self.mode = mode

        # Dynamically import the appropriate tqdm implementation
        from tqdm.auto import tqdm as tqdm_impl

        self.tqdm_impl = tqdm_impl

        # Store the actual total counts for epochs and batches
        self.total_epochs_raw = total_epochs
        self.total_batches_raw = total_batches

        # Determine the visual granularity for the progress bars
        self.g_epochs = min(g_epochs or total_epochs, total_epochs)
        self.g_batches = min(g_batches or total_batches, total_batches)

        # Set the total steps for the progress bars based on granularity
        self.total_epochs = self.g_epochs
        self.total_batches = self.g_batches

        # Initialize the tqdm progress bar instances based on the mode
        if self.mode == "train":
            # Outer bar for epochs
            self.epoch_bar = self.tqdm_impl(
                total=self.total_epochs, desc="Current Epoch", position=0, leave=True
            )
            # Inner bar for batches
            self.batch_bar = self.tqdm_impl(
                total=self.total_batches, desc="Current Batch", position=1, leave=False
            )
        elif self.mode == "eval":
            # No epoch bar needed for evaluation
            self.epoch_bar = None
            # A single bar for evaluation progress
            self.batch_bar = self.tqdm_impl(
                total=self.total_batches, desc="Evaluating", position=0, leave=False
            )

        # Keep track of the last updated visual step to avoid redundant updates
        self.last_epoch_step = -1
        self.last_batch_step = -1

        # Store the frequency for logging messages
        self.epoch_message_freq = epoch_message_freq
        self.batch_message_freq = batch_message_freq

    def update_epoch(self, epoch, postfix_dict=None):
        """
        Updates the epoch progress bar and resets the batch bar.

        Args:
            epoch: The current epoch number (1-indexed).
            postfix_dict: An optional dictionary of metrics to display
                          on the epoch bar.
        """
        # Calculate the visual step based on the current epoch and granularity
        epoch_step = math.floor((epoch - 1) * self.g_epochs / self.total_epochs_raw)

        # Update the bar only when the visual step changes
        if epoch_step != self.last_epoch_step:
            self.epoch_bar.update(1)
            self.last_epoch_step = epoch_step
        # Ensure the bar completes on the final epoch
        elif epoch == self.total_epochs_raw and self.epoch_bar.n < self.g_epochs:
            self.epoch_bar.update(1)
            self.last_epoch_step = epoch_step

        # Update the description and postfix for the epoch bar in train mode
        if self.mode == "train":
            self.epoch_bar.set_description(f"Training - Current Epoch: {epoch}")
        if postfix_dict:
            self.epoch_bar.set_postfix(postfix_dict)

        # Reset the batch bar for the new epoch
        self.batch_bar.reset()
        self.last_batch_step = -1

    def update_batch(self, batch, postfix_dict=None):
        """
        Updates the batch progress bar.

        Args:
            batch: The current batch number (1-indexed).
            postfix_dict: An optional dictionary of metrics to display
                          on the batch bar.
        """
        # Calculate the visual step based on the current batch and granularity
        batch_step = math.floor((batch - 1) * self.g_batches / self.total_batches_raw)

        # Update the bar only when the visual step changes
        if batch_step != self.last_batch_step:
            self.batch_bar.update(1)
            self.last_batch_step = batch_step
        # Ensure the bar completes on the final batch
        elif batch == self.total_batches_raw and self.batch_bar.n < self.g_batches:
            self.batch_bar.update(1)
            self.last_batch_step = batch_step

        # Update the description of the batch bar based on the mode
        if self.mode == "train":
            self.batch_bar.set_description(f"Training - Current Batch: {batch}")
        elif self.mode == "eval":
            self.batch_bar.set_description(f"Evaluation - Current Batch: {batch}")

        # Set any provided metrics on the batch bar
        if postfix_dict:
            self.batch_bar.set_postfix(postfix_dict)

    def maybe_log_epoch(self, epoch, message):
        """
        Prints a message at a specified epoch frequency.

        Args:
            epoch: The current epoch number.
            message: The message to print.
        """
        # Check if logging is enabled and if the current epoch is a logging interval
        if self.epoch_message_freq and epoch % self.epoch_message_freq == 0:
            print(message)

    def maybe_log_batch(self, batch, message):
        """
        Prints a message at a specified batch frequency.

        Args:
            batch: The current batch number.
            message: The message to print.
        """
        # Check if logging is enabled and if the current batch is a logging interval
        if self.batch_message_freq and batch % self.batch_message_freq == 0:
            print(message)

    def close(self, last_message=None):
        """
        Closes the progress bars and optionally prints a final message.

        Args:
            last_message: An optional final message to print after closing
                          the bars.
        """
        # Close the epoch bar if it exists (in 'train' mode)
        if self.mode == "train":
            self.epoch_bar.close()
        # Close the batch bar
        self.batch_bar.close()

        # Print a final message if one is provided
        if last_message:
            print(last_message)
            


def evaluate_accuracy(model, data_loader, device):
    """
    Calculates the accuracy of a model on a given dataset.

    This function iterates through the provided data loader, performs a
    forward pass with the model, and compares the predicted labels to the
    true labels to compute the overall accuracy. It operates in evaluation
    mode and disables gradient calculations for efficiency.

    Args:
        model: The neural network model to be evaluated.
        data_loader: The DataLoader providing the evaluation dataset.
        device: The device (e.g., 'cpu' or 'cuda') to run the evaluation on.

    Returns:
        The accuracy of the model on the dataset as a float.
    """
    # Initialize a progress bar for the evaluation process
    pbar = NestedProgressBar(
        total_epochs=1,
        total_batches=len(data_loader),
        mode="eval",
    )

    # Set the model to evaluation mode
    model.eval()
    # Initialize counters for correct predictions and total samples
    total_correct = 0
    total_samples = 0

    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Iterate over the batches in the data loader
        for batch_idx, (inputs, labels) in enumerate(data_loader):

            # Update the progress bar for the current batch
            pbar.update_batch(batch_idx + 1)

            # Move the input and label tensors to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            # Perform a forward pass to get the model's outputs
            outputs = model(inputs)

            # Get the predicted class by finding the index of the maximum logit
            _, predicted = outputs.max(1)
            # Tally the number of correct predictions in the batch
            total_correct += (predicted == labels).sum().item()
            # Tally the total number of samples in the batch
            total_samples += labels.size(0)

    # Close the progress bar and display a completion message
    pbar.close(last_message="Evaluation complete.")

    # Calculate the final accuracy
    accuracy = total_correct / total_samples
    # Return the computed accuracy
    return accuracy



def get_dataset_dataloaders(batch_size=64, subset_size=10_000, imbalanced=False):
    """
    Creates training and validation DataLoaders for the CIFAR-10 dataset.

    This function prepares the CIFAR-10 dataset by applying standard
    transformations. It can load the original dataset or a pre-made
    imbalanced version from a local directory. It also supports creating a
    smaller subset of the data for quicker training and splits it into
    training and validation sets.

    Args:
        batch_size: The number of samples per batch in the DataLoaders.
        subset_size: The total number of images to use from the dataset.
                     If None, the entire dataset is used.
        imbalanced: A boolean flag to switch between the standard CIFAR-10
                    dataset and a custom imbalanced version.

    Returns:
        A tuple containing the training DataLoader and the validation DataLoader.
    """
    # Define the transformation pipeline for the images
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Conditionally load the dataset based on the 'imbalanced' flag
    if imbalanced:
        # Load a custom, imbalanced dataset from a local folder
        full_trainset = ImageFolder(
            root="./cifar10_3class_imbalanced", transform=transform
        )
    else:
        # Download and load the standard CIFAR-10 training dataset
        full_trainset = datasets.CIFAR10(
            root="./cifar10", train=True, download=True, transform=transform
        )

    # Use the entire dataset if no subset size is specified
    if subset_size is None:
        subset_size = len(full_trainset)

    # Define the sizes for the training and validation splits (80-20 ratio)
    train_size = int(0.8 * subset_size)
    val_size = subset_size - train_size

    # Create a random subset of the full dataset
    subset, _ = torch.utils.data.random_split(
        full_trainset, [subset_size, len(full_trainset) - subset_size]
    )
    # Split the subset into training and validation sets
    train_subset, val_subset = random_split(subset, [train_size, val_size])

    # Create a DataLoader for the training set with shuffling enabled
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    # Create a DataLoader for the validation set without shuffling
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    # Return the configured training and validation data loaders
    return trainloader, valloader



def get_apples_dataset_dataloaders(batch_size=64, img_size=32):
    """
    Creates training and validation DataLoaders for a custom apple dataset.

    This function loads an image dataset from a specified local directory,
    applies a series of transformations (resize, convert to tensor, normalize),
    and then splits the dataset into an 80% training set and a 20%
    validation set.

    Args:
        batch_size: The number of images in each batch.
        img_size: The target height and width to which all images are resized.

    Returns:
        A tuple containing the training DataLoader and the validation DataLoader.
    """
    # Specify the local path to the root directory of the image dataset
    path_dataset = "./apple_ad_subset"

    # Define the image transformation pipeline
    transform = transforms.Compose(
        [
            # Resize images to a consistent size
            transforms.Resize((img_size, img_size)),
            # Convert images to PyTorch tensors
            transforms.ToTensor(),
            # Normalize tensor values to a range of [-1, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Load the entire dataset from the folder using the defined transformations
    full_trainset = ImageFolder(root=path_dataset, transform=transform)
    # Print the class-to-index mapping for reference
    print(full_trainset.class_to_idx)

    # Calculate the size of the training set (80% of the full dataset)
    train_size = int(0.8 * len(full_trainset))
    # Calculate the size of the validation set (the remaining 20%)
    val_size = len(full_trainset) - train_size

    # Randomly split the full dataset into training and validation subsets
    train_subset, val_subset = random_split(full_trainset, [train_size, val_size])

    # Create a DataLoader for the training set with shuffling enabled
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    # Create a DataLoader for the validation set without shuffling
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    # Return the configured training and validation data loaders
    return trainloader, valloader



def evaluate_metrics(model, test_dataloader, device, num_classes=10):
    """
    Evaluates a model's performance using multiple classification metrics.

    This function calculates the accuracy, macro-averaged precision, recall,
    and F1-score for a given model on a test dataset. It leverages the
    torchmetrics library for efficient and accurate metric computation.

    Args:
        model: The trained neural network model to be evaluated.
        test_dataloader: The DataLoader containing the test dataset.
        device: The device (e.g., 'cpu' or 'cuda') to run the evaluation on.
        num_classes: The total number of classes in the dataset.

    Returns:
        A tuple containing the computed accuracy, precision, recall, and
        F1-score as float values.
    """
    # Set the model to evaluation mode, which disables layers like dropout
    model.eval()

    # Initialize metric objects from the torchmetrics library
    accuracy_metric = torchmetrics.Accuracy(
        task="multiclass", num_classes=num_classes
    ).to(device)
    precision_metric = torchmetrics.Precision(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)
    recall_metric = torchmetrics.Recall(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)
    f1_metric = torchmetrics.F1Score(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)

    # Disable gradient calculations to save memory and computations
    with torch.no_grad():
        # Iterate over all batches in the test dataloader
        for inputs, labels in test_dataloader:

            # Move input and label tensors to the specified device
            inputs, labels = inputs.to(device), labels.to(device)

            # Perform a forward pass to get the model's output logits
            outputs = model(inputs)

            # Determine the predicted class by finding the index of the max logit
            _, predicted = torch.max(outputs, 1)

            # Update the state of each metric with the predictions and true labels
            accuracy_metric.update(predicted, labels)
            precision_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)
            f1_metric.update(predicted, labels)

    # Compute the final metric values over all batches and convert to scalars
    accuracy = accuracy_metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1 = f1_metric.compute().item()

    # Return the calculated metrics
    return accuracy, precision, recall, f1



def train_epoch(model, train_dataloader, optimizer, loss_fcn, device, pbar):
    """
    Trains the model for a single epoch.

    This function iterates over the training dataset for one full pass.
    For each batch, it performs a forward pass, computes the loss,
    executes a backward pass to calculate gradients, and updates the
    model's weights using the optimizer. It also calculates the running
    loss and accuracy for the epoch.

    Args:
        model: The neural network model to be trained.
        train_dataloader: The DataLoader providing the training data.
        optimizer: The optimization algorithm (e.g., Adam, SGD).
        loss_fcn: The loss function (e.g., CrossEntropyLoss).
        device: The device ('cpu' or 'cuda') to perform training on.
        pbar: An instance of a progress bar handler to display progress.

    Returns:
        A tuple containing the average loss and average accuracy for the epoch.
    """
    # Set the model to training mode
    model.train()
    # Initialize variables to track loss and accuracy
    running_loss = 0.0
    correct = 0
    total = 0

    # Iterate over the batches of the training data
    for batch_idx, (inputs, labels) in enumerate(train_dataloader):
        # Update the progress bar for the current batch
        pbar.update_batch(batch_idx + 1)

        # Move the inputs and labels to the specified computation device
        inputs, labels = inputs.to(device), labels.to(device)
        # Clear the gradients from the previous iteration
        optimizer.zero_grad()
        # Perform a forward pass to get the model's predictions
        outputs = model(inputs)
        # Calculate the loss between the predictions and the true labels
        loss = loss_fcn(outputs, labels)
        # Perform a backward pass to compute gradients
        loss.backward()
        # Update the model's weights based on the computed gradients
        optimizer.step()

        # Accumulate the total loss for the epoch
        running_loss += loss.item() * inputs.size(0)
        # Get the predicted classes by finding the index of the max logit
        _, predicted = outputs.max(1)
        # Update the total number of samples processed
        total += labels.size(0)
        # Update the total number of correct predictions
        correct += predicted.eq(labels).sum().item()

    # Calculate the average loss for the epoch
    epoch_loss = running_loss / total
    # Calculate the average accuracy for the epoch
    epoch_acc = correct / total

    # Return the epoch's average loss and accuracy
    return epoch_loss, epoch_acc



def train_model(model, optimizer, loss_fcn, train_dataloader, device, n_epochs):
    """
    Orchestrates the training of a model for a specified number of epochs.

    This function manages the overall training loop. It initializes a progress
    bar to visualize progress and, for each epoch, calls a helper function
    to perform the actual training steps. It also handles periodic logging
    of the training loss.

    Args:
        model: The neural network model to be trained.
        optimizer: The optimization algorithm (e.g., Adam, SGD).
        loss_fcn: The loss function used for training.
        train_dataloader: The DataLoader providing the training data.
        device: The device ('cpu' or 'cuda') to perform training on.
        n_epochs: The total number of epochs to train the model.
    """
    # Initialize a progress bar to visualize the training process
    pbar = NestedProgressBar(
        total_epochs=n_epochs,
        total_batches=len(train_dataloader),
        epoch_message_freq=5,
        mode="train",
    )

    # Begin the main training loop over the specified number of epochs
    for epoch in range(n_epochs):
        # Update the epoch-level progress bar
        pbar.update_epoch(epoch + 1)

        # Call the helper function to train the model for one epoch
        train_loss, _ = train_epoch(
            model, train_dataloader, optimizer, loss_fcn, device, pbar
        )

        # Log the training loss for the current epoch at specified intervals
        pbar.maybe_log_epoch(
            epoch + 1,
            message=f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}",
        )

    # Close the progress bar and print a final message upon completion
    pbar.close("Training complete!\n")
