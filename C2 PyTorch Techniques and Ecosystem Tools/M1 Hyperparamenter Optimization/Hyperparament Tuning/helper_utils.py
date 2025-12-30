import math
from typing import Optional

import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import torch
# from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


# = DLAI plots style =
def apply_dlai_style():
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

    # Custom colors (reusable)
    color_map = {
        "pink": "#F65B66",
        "blue": "#1C74EB",
        "yellow": "#FAB901",
        "red": "#DD3C66",
        "purple": "#A12F9D",
        "cyan": "#237B94",
    }
    return color_map, PLOT_STYLE



color_map, PLOT_STYLE = apply_dlai_style()
mpl.rcParams.update(PLOT_STYLE)



# Custom colors (reusable)
BLUE_COLOR_TRAIN = color_map["blue"]
PINK_COLOR_TEST = color_map["pink"]



def set_seed(seed=42):
    """Sets the seed for random number generators for reproducibility.

    Args:
        seed (int, optional): The seed value to use. Defaults to 42.
    """
    # Set the seed for PyTorch on CPU
    torch.manual_seed(seed)
    # Set the seed for all available GPUs
    torch.cuda.manual_seed_all(seed)
    # Ensure that cuDNN's convolutional algorithms are deterministic
    torch.backends.cudnn.deterministic = True
    # Disable the cuDNN benchmark for reproducibility
    torch.backends.cudnn.benchmark = False

    

def get_dataset_dataloaders(batch_size=64, subset_size=10_000, imbalanced=False):
    """Prepares and returns training and validation dataloaders.

    This function loads either the standard CIFAR-10 dataset or a custom
    imbalanced dataset, creates a subset, splits it into training and
    validation sets, and returns the corresponding DataLoader objects.

    Args:
        batch_size (int, optional): The number of samples per batch. Defaults to 64.
        subset_size (int, optional): The size of the dataset subset to use. 
                                     Defaults to 10,000.
        imbalanced (bool, optional): If True, loads a custom imbalanced dataset. 
                                     If False, loads standard CIFAR-10. Defaults to False.

    Returns:
        tuple: A tuple containing the training DataLoader and the validation DataLoader.
    """
    # Define the image transformation pipeline
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Conditionally load the appropriate dataset
    if imbalanced:
        # Load a custom imbalanced dataset from a local folder
        full_trainset = ImageFolder(
            root="./cifar10_3class_imbalanced", transform=transform
        )
        # Use the full size of the imbalanced dataset
        subset_size = None
    else:
        # Load the standard CIFAR-10 training dataset, downloading if necessary
        full_trainset = datasets.CIFAR10(
            root="./cifar10", train=True, download=True, transform=transform
        )

    # Use the full dataset size if a subset size is not specified
    if subset_size is None:
        subset_size = len(full_trainset)

    # Calculate the sizes for an 80/20 train-validation split
    train_size = int(0.8 * subset_size)
    val_size = subset_size - train_size

    # Create a random subset from the full dataset
    subset, _ = torch.utils.data.random_split(
        full_trainset, [subset_size, len(full_trainset) - subset_size]
    )
    # Split the subset into training and validation sets
    train_subset, val_subset = random_split(subset, [train_size, val_size])

    # Create a DataLoader for the training set with shuffling
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    # Create a DataLoader for the validation set without shuffling
    valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    return trainloader, valloader



def plot_metrics_vs_learning_rate(df_metrics):
    """Generates a scatter plot of performance metrics versus learning rates.

    Args:
        df_metrics (pd.DataFrame): A pandas DataFrame containing the results. 
                                   It must have a 'learning_rate' column and 
                                   columns for each metric to be plotted.
    """
    # Create a new figure for the plot with a specified size
    plt.figure(figsize=(10, 6))
    
    # Iterate through the metrics and plot each one against the learning rate
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        # Create a scatter plot for the current metric
        plt.scatter(
            df_metrics["learning_rate"],
            df_metrics[metric],
            marker="o",
            label=metric,
        )

    # Set the x-axis to a logarithmic scale
    plt.xscale("log")
    # Set the label for the x-axis
    plt.xlabel("Learning Rate (log scale)")
    # Set the label for the y-axis
    plt.ylabel("Metric Value")
    # Set the title of the plot
    plt.title("Metrics vs Learning Rate")
    # Display the legend to identify each metric's points
    plt.legend()
    # Enable the grid for better readability
    plt.grid(True)

    

def plot_metrics_vs_batch_size(df_metrics):
    """Generates a scatter plot of performance metrics versus batch sizes.

    Args:
        df_metrics (pd.DataFrame): A pandas DataFrame containing the results.
                                   It must have a 'batch_size' column and
                                   columns for each metric to be plotted.
    """
    # Create a new figure for the plot with a specified size
    plt.figure(figsize=(10, 6))
    
    # Iterate through the metrics and plot each one against the batch size
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        # Create a scatter plot for the current metric
        plt.scatter(
            df_metrics["batch_size"],
            df_metrics[metric],
            marker="o",
            label=metric,
        )

    # Set the label for the x-axis
    plt.xlabel("Batch Size")
    # Set the label for the y-axis
    plt.ylabel("Metric Value")
    # Set the title of the plot
    plt.title("Metrics vs Batch Size")
    # Display the legend to identify each metric's points
    plt.legend()
    # Enable the grid for better readability
    plt.grid(True)
    
    

def plot_results(learning_rates, accuracies):
    """Generates and displays a scatter plot of validation accuracy versus learning rate.

    Args:
        learning_rates (list): A list of learning rates to be plotted on the x-axis.
        accuracies (list): A list of corresponding validation accuracies for the y-axis.
    """
    # Create a new figure for the plot with a specified size
    plt.figure(figsize=(8, 6))
    # Create a scatter plot of the results
    plt.scatter(learning_rates, accuracies, marker="o", color=BLUE_COLOR_TRAIN)
    # Set the x-axis to a logarithmic scale for better visualization
    plt.xscale("log")
    # Set the label for the x-axis
    plt.xlabel("Learning Rate (log scale)")
    # Set the label for the y-axis
    plt.ylabel("Validation Accuracy")
    # Set the title of the plot
    plt.title("Learning Rate vs Validation Accuracy")
    # Enable the grid for better readability
    plt.grid(True)
    # Display the final plot
    plt.show()
    


class NestedProgressBar:
    """A handler for nested tqdm progress bars for training and evaluation loops.

    This class creates and manages an outer progress bar for epochs and an
    inner progress bar for batches. It supports both terminal and Jupyter

    notebook environments and includes a granularity feature to control the
    number of visual updates for very long processes.
    """
    def __init__(
        self,
        total_epochs,
        total_batches,
        g_epochs=None,
        g_batches=None,
        use_notebook=True,
        epoch_message_freq=None,
        batch_message_freq=None,
        mode="train",
    ):
        """Initializes the nested progress bars.

        Args:
            total_epochs (int): The absolute total number of epochs.
            total_batches (int): The absolute total number of batches per epoch.
            g_epochs (int, optional): The visual granularity for the epoch bar.
                                      Defaults to total_epochs.
            g_batches (int, optional): The visual granularity for the batch bar.
                                       Defaults to total_batches.
            use_notebook (bool, optional): If True, uses the notebook-compatible
                                           tqdm implementation. Defaults to True.
            epoch_message_freq (int, optional): Frequency to log epoch
                                                messages. Defaults to None.
            batch_message_freq (int, optional): Frequency to log batch
                                                messages. Defaults to None.
            mode (str, optional): The operational mode, either 'train' or 'eval'.
                                  Defaults to "train".
        """
        self.mode = mode

        # Select the tqdm implementation
        from tqdm.auto import tqdm as tqdm_impl

        self.tqdm_impl = tqdm_impl

        # Store the absolute total counts for epochs and batches
        self.total_epochs_raw = total_epochs
        self.total_batches_raw = total_batches

        # Determine the visual granularity, ensuring it doesn't exceed the total count
        self.g_epochs = min(g_epochs or total_epochs, total_epochs)
        self.g_batches = min(g_batches or total_batches, total_batches)

        # Set the progress bar totals to the calculated granularity
        self.total_epochs = self.g_epochs
        self.total_batches = self.g_batches

        # Initialize the tqdm progress bars based on the operational mode
        if self.mode == "train":
            self.epoch_bar = self.tqdm_impl(
                total=self.total_epochs, desc="Current Epoch", position=0, leave=True
            )
            self.batch_bar = self.tqdm_impl(
                total=self.total_batches, desc="Current Batch", position=1, leave=False
            )
        elif self.mode == "eval":
            self.epoch_bar = None
            self.batch_bar = self.tqdm_impl(
                total=self.total_batches, desc="Evaluating", position=0, leave=False
            )

        # Initialize trackers for the last visualized update step
        self.last_epoch_step = -1
        self.last_batch_step = -1

        # Store the frequency settings for logging messages
        self.epoch_message_freq = epoch_message_freq
        self.batch_message_freq = batch_message_freq

    def update_epoch(self, epoch, postfix_dict=None, message=None):
        """Updates the epoch-level progress bar.

        Args:
            epoch (int): The current epoch number.
            postfix_dict (dict, optional): A dictionary of metrics to display.
                                           Defaults to None.
            message (str, optional): A message to potentially log. Defaults to None.
        """
        # Map the raw epoch count to its corresponding visual step based on granularity
        epoch_step = math.floor((epoch - 1) * self.g_epochs / self.total_epochs_raw)

        # Update the progress bar only when the visual step changes
        if epoch_step != self.last_epoch_step:
            self.epoch_bar.update(1)
            self.last_epoch_step = epoch_step
        # Ensure the progress bar completes on the final epoch
        elif epoch == self.total_epochs_raw and self.epoch_bar.n < self.g_epochs:
            self.epoch_bar.update(1)
            self.last_epoch_step = epoch_step

        # Set the dynamic description for the progress bar
        if self.mode == "train":
            self.epoch_bar.set_description(f"Training - Current Epoch: {epoch}")
        # Update the postfix with any provided metrics or information
        if postfix_dict:
            self.epoch_bar.set_postfix(postfix_dict)

        # Reset the inner batch bar at the start of each new epoch
        self.batch_bar.reset()
        self.last_batch_step = -1

    def update_batch(self, batch, postfix_dict=None, message=None):
        """Updates the batch-level progress bar.

        Args:
            batch (int): The current batch number.
            postfix_dict (dict, optional): A dictionary of metrics to display.
                                           Defaults to None.
            message (str, optional): A message to potentially log. Defaults to None.
        """
        # Map the raw batch count to its corresponding visual step
        batch_step = math.floor((batch - 1) * self.g_batches / self.total_batches_raw)

        # Update the progress bar only when the visual step changes
        if batch_step != self.last_batch_step:
            self.batch_bar.update(1)
            self.last_batch_step = batch_step
        # Ensure the progress bar completes on the final batch
        elif batch == self.total_batches_raw and self.batch_bar.n < self.g_batches:
            self.batch_bar.update(1)
            self.last_batch_step = batch_step

        # Set the dynamic description for the progress bar based on the mode
        if self.mode == "train":
            self.batch_bar.set_description(f"Training - Current Batch: {batch}")
        elif self.mode == "eval":
            self.batch_bar.set_description(f"Evaluation - Current Batch: {batch}")

        # Update the postfix with any provided metrics
        if postfix_dict:
            self.batch_bar.set_postfix(postfix_dict)

    def maybe_log_epoch(self, epoch, message):
        """Logs a message at a specified epoch frequency.

        Args:
            epoch (int): The current epoch number.
            message (str): The message to log.
        """
        if self.epoch_message_freq and epoch % self.epoch_message_freq == 0:
            print(message)

    def maybe_log_batch(self, batch, message):
        """Logs a message at a specified batch frequency.

        Args:
            batch (int): The current batch number.
            message (str): The message to log.
        """
        if self.batch_message_freq and batch % self.batch_message_freq == 0:
            print(message)

    def close(self, last_message=None):
        """Closes all active progress bars and optionally prints a final message.

        Args:
            last_message (str, optional): A final message to print after closing.
                                          Defaults to None.
        """
        # Close the outer epoch bar if it exists (in training mode)
        if self.mode == "train":
            self.epoch_bar.close()
        # Close the inner batch bar
        self.batch_bar.close()

        # Print a concluding message if one is provided
        if last_message:
            print(last_message)


            
def train_epoch(model, train_dataloader, optimizer, loss_fcn, device, pbar):
    """Trains the model for a single epoch.

    This function iterates over the training dataloader, performs the forward
    and backward passes, updates the model weights, and calculates the loss
    and accuracy for the entire epoch.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_dataloader (DataLoader): The DataLoader containing the training data.
        optimizer (optim.Optimizer): The optimizer for updating model weights.
        loss_fcn: The loss function used for training.
        device: The device (e.g., 'cuda' or 'cpu') to perform training on.
        pbar: A progress bar handler object to visualize training progress.

    Returns:
        tuple: A tuple containing the average loss and accuracy for the epoch.
    """
    # Set the model to training mode
    model.train()
    # Initialize metrics for the epoch
    running_loss = 0.0
    correct = 0
    total = 0

    # Iterate over the training data
    for batch_idx, (inputs, labels) in enumerate(train_dataloader):
        # Update the batch progress bar
        pbar.update_batch(batch_idx + 1)

        # Move input and label tensors to the specified device
        inputs, labels = inputs.to(device), labels.to(device)
        # Clear the gradients from the previous iteration
        optimizer.zero_grad()
        # Perform a forward pass to get model outputs
        outputs = model(inputs)
        # Calculate the loss
        loss = loss_fcn(outputs, labels)
        # Perform a backward pass to compute gradients
        loss.backward()
        # Update the model's weights
        optimizer.step()

        # Accumulate the loss for the epoch
        running_loss += loss.item() * inputs.size(0)
        # Get the predicted class with the highest score
        _, predicted = outputs.max(1)
        # Update the total number of samples
        total += labels.size(0)
        # Update the number of correctly classified samples
        correct += predicted.eq(labels).sum().item()

    # Calculate the average loss for the epoch
    epoch_loss = running_loss / total
    # Calculate the average accuracy for the epoch
    epoch_acc = correct / total

    return epoch_loss, epoch_acc



def train_model(model, optimizer, loss_fcn, train_dataloader, device, n_epochs):
    """Coordinates the training process for a model over multiple epochs.

    This function sets up a progress bar and manages the training loop,
    calling a helper function to handle the logic for each individual epoch.
    It also logs progress periodically.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        optimizer (optim.Optimizer): The optimizer for updating model weights.
        loss_fcn: The loss function used for training.
        train_dataloader (DataLoader): The DataLoader for the training data.
        device: The device (e.g., 'cuda' or 'cpu') to perform training on.
        n_epochs (int): The total number of epochs to train for.
    """
    # Initialize the nested progress bar for visualizing the training process
    pbar = NestedProgressBar(
        total_epochs=n_epochs,
        total_batches=len(train_dataloader),
        epoch_message_freq=5,
        mode="train",
        use_notebook=True,
    )

    # Loop through the specified number of training epochs
    for epoch in range(n_epochs):
        # Update the outer progress bar for the current epoch
        pbar.update_epoch(epoch + 1)

        # Call the helper function to train the model for one full epoch
        train_loss, _ = train_epoch(
            model, train_dataloader, optimizer, loss_fcn, device, pbar
        )

        # Log the training loss for the current epoch at a set frequency
        pbar.maybe_log_epoch(
            epoch + 1,
            message=f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}",
        )

    # Close the progress bar and print a final completion message
    pbar.close("Training complete!\n")