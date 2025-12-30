import math
import time
from functools import wraps
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm



# Global plot style
PLOT_STYLE = {
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "font.family": "sans", # "sans-serif",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 3,
    "lines.markersize": 6,
}

plt.rcParams.update(PLOT_STYLE)

# Custom colors (reusable)
PINK = "#F65B66"  # Pink
BLUE = "#237B94"  # Blue
DARK_BLUE = "#1C74EB"# Dark Blue
YELLOW = "#FAB901" # Yellow
PURPLE = "#A12F9D" # Purple



class NestedProgressBar:
    """Manages nested tqdm progress bars for training and evaluation loops.

    Args:
        total_epochs: The total number of epochs for the entire process.
        total_batches: The total number of batches within a single epoch.
        g_epochs: The granularity or number of visible steps for the epoch bar.
                  If None, it defaults to total_epochs.
        g_batches: The granularity or number of visible steps for the batch bar.
                   If None, it defaults to total_batches.
        use_notebook: Determines whether to use the notebook-friendly version of tqdm.
        epoch_message_freq: The frequency (in epochs) at which to log messages.
        batch_message_freq: The frequency (in batches) at which to log messages.
        mode: The operational mode, either 'train' or 'eval'.
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

        self.mode = mode

        # Select the appropriate tqdm implementation
        from tqdm.auto import tqdm as tqdm_impl
        self.tqdm_impl = tqdm_impl

        # Store the total counts for calculation purposes
        self.total_epochs_raw = total_epochs
        self.total_batches_raw = total_batches

        # Set the granularity, ensuring it does not exceed the total count
        self.g_epochs = min(g_epochs or total_epochs, total_epochs)
        self.g_batches = min(g_batches or total_batches, total_batches)
        self.total_epochs = self.g_epochs
        self.total_batches = self.g_batches

        # Initialize the tqdm progress bars based on the specified mode
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

        # Keep track of the last updated step to manage granularity
        self.last_epoch_step = -1
        self.last_batch_step = -1

        # Store message frequency settings for logging
        self.epoch_message_freq = epoch_message_freq
        self.batch_message_freq = batch_message_freq

    def update_epoch(self, epoch, postfix_dict=None, message=None):
        """Updates the epoch progress bar and resets the batch bar.

        Args:
            epoch: The current epoch number (1-based).
            postfix_dict: A dictionary of metrics to display on the progress bar.
            message: An optional message to be logged for this epoch.
        """
        # Map the raw epoch to its corresponding visual step on the progress bar
        epoch_step = math.floor((epoch - 1) * self.g_epochs / self.total_epochs_raw)

        # Update the progress bar only when a new visual step is reached
        if epoch_step != self.last_epoch_step:
            self.epoch_bar.update(1)
            self.last_epoch_step = epoch_step
        # Ensure the progress bar completes on the final epoch
        elif epoch == self.total_epochs_raw and self.epoch_bar.n < self.g_epochs:
            self.epoch_bar.update(1)
            self.last_epoch_step = epoch_step

        # Set the display information for the progress bar
        if self.mode == "train":
            self.epoch_bar.set_description(f"Training - Current Epoch: {epoch}")
        if postfix_dict:
            self.epoch_bar.set_postfix(postfix_dict)

        # Reset the batch bar for the new epoch
        self.batch_bar.reset()
        self.last_batch_step = -1

    def update_batch(self, batch, postfix_dict=None, message=None):
        """Updates the batch progress bar.

        Args:
            batch: The current batch number (1-based).
            postfix_dict: A dictionary of metrics to display on the progress bar.
            message: An optional message to be logged for this batch.
        """
        # Map the raw batch to its corresponding visual step on the progress bar
        batch_step = math.floor((batch - 1) * self.g_batches / self.total_batches_raw)
        
        # Update the progress bar only when a new visual step is reached
        if batch_step != self.last_batch_step:
            self.batch_bar.update(1)
            self.last_batch_step = batch_step
        # Ensure the progress bar completes on the final batch
        elif batch == self.total_batches_raw and self.batch_bar.n < self.g_batches:
            self.batch_bar.update(1)
            self.last_batch_step = batch_step

        # Set the display information for the progress bar based on the mode
        if self.mode == "train":
            self.batch_bar.set_description(f"Training - Current Batch: {batch}")
        elif self.mode == "eval":
            self.batch_bar.set_description(f"Evaluation - Current Batch: {batch}")

        if postfix_dict:
            self.batch_bar.set_postfix(postfix_dict)

    def maybe_log_epoch(self, epoch, message):
        """Logs a message at a specified epoch frequency.

        Args:
            epoch: The current epoch number.
            message: The message to log.
        """
        if self.epoch_message_freq and epoch % self.epoch_message_freq == 0:
            print(message)

    def maybe_log_batch(self, batch, message):
        """Logs a message at a specified batch frequency.

        Args:
            batch: The current batch number.
            message: The message to log.
        """
        if self.batch_message_freq and batch % self.batch_message_freq == 0:
            print(message)

    def close(self, last_message=None):
        """Closes the progress bars and prints a final message if provided.

        Args:
            last_message: An optional final message to print after closing.
        """
        # Close the epoch bar only if it exists (i.e., in 'train' mode)
        if self.mode == "train":
            self.epoch_bar.close()
        # Close the batch bar in all modes
        self.batch_bar.close()
        
        # Print a final message if one is provided
        if last_message:
            print(last_message)


            
def load_resnet_table():
    """Loads ResNet model performance data from a CSV file.

    This function reads a specific CSV file named 'resnet_results.csv'
    and returns its contents as a pandas DataFrame.

    Returns:
        A pandas DataFrame containing the ResNet results.
    """
    # Read the CSV file into a pandas DataFrame, using the first column as the index.
    resnet_results = pd.read_csv("resnet_results.csv", index_col=0)
    # Return the loaded DataFrame.
    return resnet_results



class PbarEpoch:
    """
    Manages a tqdm progress bar for a single training epoch.

    This class encapsulates the creation, updating, and closing of a progress bar,
    making it easier to monitor training progress on a per-epoch basis.
    """
    def __init__(self, train_loader, steps, epoch):
        """
        Initializes the progress bar for the epoch.

        Args:
            train_loader: The DataLoader for the training dataset.
            steps: The total number of steps to display on the progress bar.
            epoch: The current epoch number for the description.
        """
        # Calculate how many batches correspond to a single step in the progress bar.
        self.batches_per_step = len(train_loader) // steps
        # Initialize the tqdm progress bar with its total steps and description.
        self.pbar = tqdm(total=steps, desc=f"Train Epoch {epoch}")

    def update(self, batch_idx, loss):
        """
        Updates the progress bar with the latest batch information.

        Args:
            batch_idx: The index of the current batch being processed.
            loss: The loss value from the current batch.
        """
        # Advance the progress bar by one step.
        self.pbar.update(1)
        # Set the postfix text to display the current loss and batch number.
        self.pbar.set_postfix(current_loss=loss.item(), batch=batch_idx + 1)
        # Print the current loss and batch number to the console.
        print(f"Current Loss: {loss.item():.4f}, Batch: {batch_idx + 1}")

    def close(self):
        """
        Closes the progress bar at the end of the epoch.
        """
        # Finalize and close the tqdm progress bar instance.
        self.pbar.close()
        


def train_epoch(model, train_loader, optimizer, loss_fcn, device, pbar):
    """Performs a single training epoch for a given model.

    Args:
        model: The neural network model to be trained.
        train_loader: The DataLoader providing the training data.
        optimizer: The optimization algorithm (e.g., Adam, SGD).
        loss_fcn: The loss function used for training.
        device: The device to run the training on ('cpu' or 'cuda').
        pbar: A progress bar object to visualize training progress.

    Returns:
        A tuple containing the average loss and accuracy for the epoch.
    """
    # Set the model to training mode.
    model.train()
    # Initialize the total loss for the epoch.
    running_loss = 0.0
    # Initialize the count of correctly classified samples.
    correct = 0
    # Initialize the total number of samples processed.
    total = 0

    # Iterate over each batch of data in the training loader.
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # Update the batch progress bar.
        pbar.update_batch(batch_idx + 1)

        # Move input data and labels to the specified device.
        inputs, labels = inputs.to(device), labels.to(device)
        # Clear the gradients from the previous iteration.
        optimizer.zero_grad()
        # Perform a forward pass to get model predictions.
        outputs = model(inputs)
        # Calculate the loss between predictions and actual labels.
        loss = loss_fcn(outputs, labels)
        # Perform a backward pass to compute gradients.
        loss.backward()
        # Update the model's weights based on the gradients.
        optimizer.step()

        # Accumulate the loss for the epoch.
        running_loss += loss.item() * inputs.size(0)
        # Get the index of the max log-probability for the predictions.
        _, predicted = outputs.max(1)
        # Update the total count of processed samples.
        total += labels.size(0)
        # Update the count of correctly classified samples.
        correct += predicted.eq(labels).sum().item()
    
    # Calculate the average loss over all samples in the epoch.
    epoch_loss = running_loss / total
    # Calculate the accuracy for the epoch.
    epoch_acc = correct / total

    # Return the calculated epoch loss and accuracy.
    return epoch_loss, epoch_acc


def evaluate_accuracy(model, data_loader, device):
    """Calculates the accuracy of a model on a given dataset.

    Args:
        model: The neural network model to be evaluated.
        data_loader: The DataLoader providing the evaluation data.
        device: The device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        The accuracy of the model as a float.
    """
    # Initialize a progress bar for visualizing the evaluation process.
    pbar = NestedProgressBar(
        total_epochs=1,
        total_batches=len(data_loader),
        mode="eval",
        use_notebook=True,
    )

    # Set the model to evaluation mode.
    model.eval()
    # Initialize a counter for the number of correct predictions.
    total_correct = 0
    # Initialize a counter for the total number of samples.
    total_samples = 0

    # Disable gradient calculations to speed up inference.
    with torch.no_grad():
        # Iterate over each batch in the data loader.
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            # Update the progress bar for the current batch.
            pbar.update_batch(batch_idx + 1)

            # Move the input data and labels to the specified device.
            inputs, labels = inputs.to(device), labels.to(device)
            # Perform a forward pass to get the model's predictions.
            outputs = model(inputs)

            # Get the class with the highest score as the prediction.
            _, predicted = outputs.max(1)
            # Increment the count of correct predictions.
            total_correct += (predicted == labels).sum().item()
            # Increment the total sample count.
            total_samples += labels.size(0)
    
    # Close the progress bar upon completion.
    pbar.close()

    # Calculate the final accuracy.
    accuracy = total_correct / total_samples
    # Return the computed accuracy.
    return accuracy



def plot_efficiency_analysis(results_df):
    """
    Generates and displays a scatter plot to analyze model efficiency.

    This function visualizes the relationship between model accuracy, inference time,
    and size. Each model is represented as a point, where the x-axis is the
    inference time, the y-axis is the accuracy, and the size of the point
    corresponds to the model's file size. Annotations are placed dynamically
    based on marker size.

    Args:
        results_df: A pandas DataFrame containing the performance metrics for
                    each model. It must include the columns 'inference_time_ms',
                    'accuracy', and 'model_size_mb'.
    """
    # Set up the plot figure and get its axes for more control over elements.
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define a base font size for labels.
    label_fontsize = 15 + (10 / 5)

    # Define a list of colors for the plot points.
    colors = [PINK, BLUE, DARK_BLUE, YELLOW, PURPLE]
    # Initialize a counter to cycle through the color list.
    i = 0

    # Iterate over each model's data in the DataFrame.
    for name, row in results_df.iterrows():
        # Define the marker area based on the model's size.
        marker_area = row["model_size_mb"] * 10.15
        
        # Create a scatter plot point for the current model.
        ax.scatter(
            row["inference_time_ms"],
            row["accuracy"],
            s=marker_area,
            label=str(round(row["model_size_mb"], 1)) + " MB - " + name,
            c=colors[i],
        )

        # Calculate the marker's radius in points to position the text label.
        marker_radius_pts = np.sqrt(marker_area)
        # Calculate the vertical offset to place the text just below the marker.
        vertical_offset_pts = - (marker_radius_pts * 1.02)

        # Add a text annotation showing the exact accuracy value.
        ax.annotate(
            f"{row['accuracy']:.2f}",
            # Set the annotation's anchor point to the center of the marker.
            xy=(row["inference_time_ms"], row["accuracy"]),
            # Set the text's position using an offset in points.
            xytext=(0, vertical_offset_pts),
            # Specify that the offset is measured in points.
            textcoords="offset points",
            fontsize=label_fontsize,
            ha="center",
            # Vertically align the top of the text to the offset point.
            va="top",
        )
        # Move to the next color for the next model.
        i += 1

    # Set the title of the plot.
    ax.set_title("Model Efficiency: Inference Time vs Accuracy")
    # Set the label for the x-axis.
    ax.set_xlabel("Inference Time (ms)")
    # Set the label for the y-axis.
    ax.set_ylabel("Accuracy")
    
    # Dynamically calculate the x-axis limit with some padding.
    xmax = results_df["inference_time_ms"].max() * 1.15
    # Apply the calculated x-axis limit.
    ax.set_xlim(0, xmax)
    # Set the y-axis limit with a small padding at the top.
    ax.set_ylim(0, 1.05)
    
    # Explicitly define the tick marks on the y-axis for consistency.
    ax.set_yticks(np.arange(0, 1.1, 0.2))

    # Create the plot's legend with the defined font size.
    legend = ax.legend(fontsize=label_fontsize)
    # Ensure all markers in the legend have a consistent size for clarity.
    for handle in legend.legend_handles:
        handle._sizes = [80]
        
    # Display a grid on the plot for better readability.
    ax.grid(True)
    # Adjust the plot to ensure all elements fit without overlapping.
    fig.tight_layout()
    # Show the final generated plot.
    plt.show()