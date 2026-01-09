import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tv_models
from IPython.display import display, HTML
from sklearn.metrics import ConfusionMatrixDisplay
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
)
from torchvision import datasets
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
    mean = torch.tensor([0.485, 0.490, 0.451])
    std = torch.tensor([0.214, 0.197, 0.191])
    
    # Create a copy of the tensor to avoid modifying the original in-place.
    unnormalized_tensor = tensor.clone()
    
    # Apply the unnormalization formula to each channel: (pixel * std) + mean.
    for i, (m, s) in enumerate(zip(mean, std)):
        unnormalized_tensor[i].mul_(s).add_(m)
        
    # Clamp pixel values to the valid [0, 1] range to correct for floating-point inaccuracies.
    unnormalized_tensor = torch.clamp(unnormalized_tensor, 0, 1)
    
    # Return the unnormalized tensor.
    return unnormalized_tensor



def create_datasets(dataset_path, train_transform, val_transform, train_split=0.8, seed=42):
    """
    Initializes and splits an image dataset from a directory structure.

    This function loads a dataset using ImageFolder, performs a random split
    to create training and validation subsets, and then applies separate
    data transformations to each. A nested class is used to wrap the subsets,
    ensuring transformations are applied on-the-fly during data loading.

    Args:
        dataset_path (str): The file path to the root of the image dataset.
        train_transform (callable): The transformations to apply to the training set.
        val_transform (callable): The transformations to apply to the validation set.
        train_split (float, optional): The proportion of the dataset to allocate
                                     to the training split. Defaults to 0.8.
        seed (int, optional): A seed for the random number generator to ensure
                              a reproducible split. Defaults to 42.

    Returns:
        tuple: A tuple containing the transformed training and validation datasets.
    """
    
    # --- Nested Class for Applying Transformations ---
    class TransformedDataset(Dataset):
        """
        A wrapper dataset that applies a given transformation to a subset.

        This allows for different transformations to be applied to datasets that
        have already been split, such as training and validation sets.

        Args:
            subset (torch.utils.data.Subset): The dataset subset to wrap.
            transform (callable): The transformation pipeline to apply to the images.
        """
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            # Inherit class attributes from the original full dataset
            self.classes = subset.dataset.classes
            self.class_to_idx = subset.dataset.class_to_idx

        def __len__(self):
            """Returns the total number of samples in the subset."""
            return len(self.subset)

        def __getitem__(self, idx):
            """
            Retrieves an image and its label from the subset and applies the
            transformation to the image.

            Returns:
                tuple: A tuple containing the transformed image and its label.
            """
            img, label = self.subset[idx]
            return self.transform(img), label

    # Load the entire dataset from the specified path without applying any transformations yet.
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=None)

    # Determine the number of samples for the training and validation sets.
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Perform a random split of the dataset using a seeded generator for reproducibility.
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Wrap the subsets with the custom TransformedDataset class to apply the appropriate transformations.
    train_dataset = TransformedDataset(subset=train_subset, transform=train_transform)
    val_dataset = TransformedDataset(subset=val_subset, transform=val_transform)

    return train_dataset, val_dataset



def create_dataloaders(train_dataset, test_dataset, batch_size):
    """
    Initializes and configures DataLoaders for training and testing datasets.

    This function wraps dataset objects into DataLoader instances, which provide
    utilities for batching, shuffling, and iterating over the data during model
    training and evaluation.

    Args:
        train_dataset (Dataset): The dataset object for training.
        test_dataset (Dataset): The dataset object for testing or validation.
        batch_size (int): The number of samples to include in each batch.

    Returns:
        tuple: A tuple containing the configured training and testing DataLoaders.
    """
    
    # Create the DataLoader for the training set with shuffling enabled.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Create the DataLoader for the testing set with shuffling disabled for consistent evaluation.
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Return the configured training and testing DataLoaders.
    return train_loader, test_loader



def show_sample_images(dataset, class_names):
    """
    Displays a grid of sample images from the dataset.

    This function creates a plot showing one randomly selected image from each
    class and uses the provided `class_names` list for the titles.

    Args:
        dataset (Dataset): The dataset to visualize. Must have a '.classes'
                           attribute and support subset indexing.
        class_names (list of str): A list of formatted class names for the plot titles.
    """
    # Get the total number of classes from the dataset.
    num_classes = len(dataset.classes)
    
    # Validate that the number of class names matches the number of classes.
    assert len(class_names) == num_classes, "Length of class_names list must match the number of classes in the dataset."

    # Create a mapping of class index to all its image indices.
    class_to_indices = {i: [] for i in range(num_classes)}
    full_dataset_targets = dataset.subset.dataset.targets
    subset_indices = dataset.subset.indices
    for subset_idx, full_idx in enumerate(subset_indices):
        label = full_dataset_targets[full_idx]
        class_to_indices[label].append(subset_idx)

    # Dynamically calculate the grid size for the plot.
    ncols = 7
    nrows = (num_classes + ncols - 1) // ncols  # Ceiling division for rows.
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, nrows * 2.2))

    # Loop through each class to display one random sample.
    for i, ax in enumerate(axes.flatten()):
        # Hide axes for any empty subplots.
        if i >= num_classes:
            ax.axis('off')
            continue

        # Set the plot title using the provided class names.
        class_name = class_names[i]

        # Pick a random image from the current class.
        random_image_idx = random.choice(class_to_indices[i])

        # Retrieve the image and label from the dataset.
        image, label = dataset[random_image_idx]

        # Un-normalize the image for proper display.
        # Assumes an 'unnormalize' function is available.
        image = unnormalize(image)

        # Prepare the image tensor for plotting.
        npimg = image.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(class_name)
        ax.axis('off')

    # Apply a tight layout and show the plot.
    plt.tight_layout()
    plt.show()
    
    
    
def display_torch_summary(summary_object, attr_names, display_names, depth):
    """
    Displays a torchinfo summary object as a styled HTML table.

    This function processes a summary object from the torchinfo library,
    formats it into a pandas DataFrame, and then renders it as a clean,
    readable HTML table within a Jupyter environment. It also displays
    key summary statistics like total parameters and memory usage below the table.

    Args:
        summary_object: The object returned by `torchinfo.summary()`.
        attr_names (list): A list of the layer attribute names to extract
                           (e.g., 'input_size', 'num_params').
        display_names (list): A list of the desired column headers for the
                              output table (e.g., 'Input Shape', 'Param #').
        depth (int, optional): The maximum depth of layers to display.
                               Defaults to infinity (showing all layers).
    """

    layer_data = []
    # Define the table column headers for the DataFrame with the new name.
    display_columns = ["Layer (type (var_name):depth-idx)"] + display_names

    for layer in summary_object.summary_list:
        # Only process layers that are within the specified depth.
        if layer.depth > depth:
            continue

        row = {}

        # Construct the hierarchical layer name with the var_name.
        indent = "&nbsp;"*4*layer.depth
        # NEW: Construct the layer name with the var_name included.
        layer_name = f"{layer.class_name} ({layer.var_name})"
        if layer.depth > 0:
            # Append depth and index for nested layers.
            layer_name = f"{layer_name}: {layer.depth}-{layer.depth_index}"

        row["Layer (type (var_name):depth-idx)"] = f"{indent}{layer_name}"

        # Iterate over both attribute and display names to populate row data.
        for attr, name in zip(attr_names, display_names):
            if attr == "num_params":
                # Mimic torchinfo's logic for displaying parameters.
                show_params = layer.is_leaf_layer or layer.depth == depth
                if show_params and layer.num_params > 0:
                    value = f"{layer.num_params:,}"
                else:
                    value = "--"
            else:
                # Fetch all other attributes directly.
                value = getattr(layer, attr, "N/A")

            row[name] = value
        layer_data.append(row)

    df = pd.DataFrame(layer_data, columns=display_columns)

    # Style the DataFrame for clean HTML presentation.
    styler = df.style.hide(axis="index")
    styler.set_table_styles([
        {"selector": "table", "props": [("width", "100%"), ("border-collapse", "collapse")]},
        {"selector": "th", "props": [
            ("text-align", "left"), ("padding", "8px"),
            ("background-color", "#4f4f4f"), ("color", "white"),
            ("border-bottom", "1px solid #ddd")
        ]},
        {"selector": "td", "props": [
            ("text-align", "left"), ("padding", "8px"),
            ("border-bottom", "1px solid #ddd")
        ]},
    ]).set_properties(**{"white-space": "pre", "vertical-align": "top"})

    table_html = styler.to_html()

    # --- Summary Statistics ---
    total_params = f"{summary_object.total_params:,}"
    trainable_params = f"{summary_object.trainable_params:,}"
    non_trainable_params = f"{summary_object.total_params - summary_object.trainable_params:,}"
    total_mult_adds = f"{summary_object.total_mult_adds/1e9:.2f} GB"

    params_html = f"""
    <div style="margin-top: 20px; font-family: monospace; line-height: 1.6;">
        <hr><p><b>Total params:</b> {total_params}</p>
        <p><b>Trainable params:</b> {trainable_params}</p>
        <p><b>Non-trainable params:</b> {non_trainable_params}</p>
        <p><b>Total mult-adds:</b> {total_mult_adds}</p><hr>
    </div>"""

    input_size_mb = summary_object.total_input/(1024**2)
    fwd_bwd_pass_size_mb = summary_object.total_output_bytes/(1024**2)
    params_size_mb = summary_object.total_param_bytes/(1024**2)
    total_size_mb = (
        summary_object.total_input +
        summary_object.total_output_bytes +
        summary_object.total_param_bytes
    )/(1024**2)

    size_html = f"""
    <div style="font-family: monospace; line-height: 1.6;">
        <p><b>Input size (MB):</b> {input_size_mb:.2f}</p>
        <p><b>Forward/backward pass size (MB):</b> {fwd_bwd_pass_size_mb:.2f}</p>
        <p><b>Params size (MB):</b> {params_size_mb:.2f}</p>
        <p><b>Estimated Total Size (MB):</b> {total_size_mb:.2f}</p><hr>
    </div>"""

    # Combine all HTML parts and display.
    final_html = table_html + params_html + size_html
    display(HTML(final_html))


def training_loop_16_mixed(
    model, train_loader, val_loader, loss_function, optimizer, num_epochs, device, scheduler=None, save_path=None,
):
    """Executes a training and validation loop for a PyTorch model using 16-bit mixed precision.

    This function iterates through epochs, performing a training step and a
    validation step for each. It tracks performance metrics like loss and
    accuracy, and optionally saves the model with the highest validation accuracy.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): The DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): The DataLoader for validation data.
        loss_function (callable): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimization algorithm.
        num_epochs (int): The total number of training epochs.
        device (torch.device): The device to perform training on (e.g., 'cuda', 'cpu').
        scheduler (torch.optim.lr_scheduler, optional): A learning rate scheduler. Defaults to None.
        save_path (str, optional): A file path to save the best model weights. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The model with the weights that achieved the best validation accuracy.
            - history (dict): A dictionary containing training and validation loss and validation accuracy for each epoch.
            - best_cm (numpy.ndarray): The confusion matrix from the best validation epoch.
    """
    # Determine the device type as a string for autocast compatibility.
    if device == torch.device("mps"):
        device_str = "mps"
    elif device == torch.device("cuda"):
        device_str = "cuda"
    else:
        device_str = "cpu"

    # Initialize a gradient scaler for mixed-precision training.
    # Gradient scaling is not supported on MPS, so it's conditionally enabled.
    use_scaler = device_str != "mps"
    scaler = GradScaler() if use_scaler else None

    # Create the directory to save the model if a path is provided.
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    # Move the model to the specified computation device.
    model.to(device)

    # Initialize variables to track the best validation accuracy and confusion matrix.
    best_val_acc = 0.0
    best_cm = None

    # A dictionary to store the history of training and validation metrics.
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # Determine the number of classes from the dataset.
    num_classes = len(train_loader.dataset.classes)
    # Initialize torchmetrics for calculating accuracy and the confusion matrix.
    val_accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro").to(
        device
    )
    val_cm = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    # Set up a single progress bar for the entire training process.
    total_steps = (len(train_loader) + len(val_loader)) * num_epochs
    pbar = tqdm(total=total_steps, desc="Overall Progress")

    # Begin the main training loop over the specified number of epochs.
    for epoch in range(num_epochs):
        # --- Training Phase ---
        # Set the model to training mode.
        model.train()
        # Initialize variables to accumulate training loss for the current epoch.
        running_train_loss = 0.0
        train_samples_processed = 0

        # Iterate over the training data loader.
        for inputs, labels in train_loader:
            # Update the progress bar description for the current phase.
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs} [Training]")
            # Move input data and labels to the designated device.
            inputs, labels = inputs.to(device), labels.to(device)
            # Clear any previously calculated gradients.
            optimizer.zero_grad(set_to_none=True)

            # Use autocast for mixed-precision forward pass.
            with autocast(device_type=device_str, dtype=torch.float16):
                # Forward pass: compute predicted outputs by passing inputs to the model.
                outputs = model(inputs)
                # Calculate the loss.
                loss = loss_function(outputs, labels)

            # If using a scaler (e.g., on CUDA), scale the loss and perform a backward pass.
            if use_scaler:
                scaler.scale(loss).backward()
                # Update the model weights using the scaled gradients.
                scaler.step(optimizer)
                # Update the scaler for the next iteration.
                scaler.update()
            # If not using a scaler (e.g., on CPU/MPS), perform a standard backward pass.
            else:
                loss.backward()
                # Update the model weights.
                optimizer.step()

            # Get the size of the current batch.
            batch_size = inputs.size(0)
            # Accumulate the loss, weighted by the batch size.
            running_train_loss += loss.item() * batch_size
            # Update the count of processed samples.
            train_samples_processed += batch_size
            # Calculate and display the running average loss.
            display_loss = running_train_loss / train_samples_processed
            pbar.set_postfix(loss=f"{display_loss:.4f}")
            # Update the progress bar for the batch.
            pbar.update(1)

        # Calculate the average training loss for the epoch.
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        # Store the epoch's training loss in the history.
        history["train_loss"].append(epoch_train_loss)

        # --- Validation Phase ---
        # Set the model to evaluation mode.
        model.eval()
        # Initialize variables to accumulate validation loss.
        running_val_loss = 0.0
        val_samples_processed = 0
        # Reset metric calculators for the new validation epoch.
        val_accuracy.reset()
        val_cm.reset()

        # Disable gradient calculations for the validation phase.
        with torch.no_grad():
            # Iterate over the validation data loader.
            for inputs, labels in val_loader:
                # Update the progress bar description for the validation phase.
                pbar.set_description(f"Epoch {epoch+1}/{num_epochs} [Validation]")
                # Move input data and labels to the designated device.
                inputs, labels = inputs.to(device), labels.to(device)

                # Use autocast for mixed-precision forward pass during validation.
                with autocast(device_type=device_str, dtype=torch.float16):
                    # Compute model outputs.
                    outputs = model(inputs)
                    # Calculate the validation loss.
                    loss = loss_function(outputs, labels)

                # Get the predicted class indices by finding the max logit.
                preds = outputs.argmax(dim=1)
                # Get the size of the current batch.
                batch_size = inputs.size(0)
                # Accumulate the validation loss.
                running_val_loss += loss.item() * batch_size
                # Update the count of processed validation samples.
                val_samples_processed += batch_size
                # Update metrics with the current batch's predictions and labels.
                val_accuracy.update(preds, labels)
                val_cm.update(preds, labels)

                # Compute and display the current running validation accuracy and loss.
                current_acc = val_accuracy.compute().item()
                display_loss = running_val_loss / val_samples_processed
                pbar.set_postfix(
                    acc=f"{current_acc:.2%}",
                    loss=f"{display_loss:.4f}",
                )
                # Update the progress bar.
                pbar.update(1)

        # Calculate the average validation loss for the epoch.
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        # Compute the final validation accuracy for the epoch.
        epoch_val_acc = val_accuracy.compute().item()
        # Store the epoch's validation loss and accuracy in the history.
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)

        # Print the summary of the epoch's performance.
        tqdm.write(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Val Loss: {epoch_val_loss:.4f}, "
            f"Val Acc: {epoch_val_acc:.4f}"
        )

        # --- SCHEDULER AND SAVE BEST MODEL ---
        # Adjust the learning rate based on the scheduler's logic, if one is provided.
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(epoch_val_acc)
            else:
                scheduler.step()

        # Check if the current model has the best validation accuracy seen so far.
        if epoch_val_acc > best_val_acc:
            # Update the best validation accuracy.
            best_val_acc = epoch_val_acc
            # Store the confusion matrix from the best performing epoch.
            best_cm = val_cm.compute().cpu().numpy()
            # Save the model's state dictionary if a save path is specified.
            if save_path:
                torch.save(model.state_dict(), save_path)
                # Print a message indicating that a new best model has been saved.
                tqdm.write(
                    f"  -> New best model saved to '{save_path}' with Val Acc: {best_val_acc:.2%}\n"
                )

    # Close the progress bar after the training loop is complete.
    pbar.close()

    # If a save path was provided, load the weights of the best performing model.
    if save_path and os.path.exists(save_path):
        tqdm.write(f"\nBest model saved to '{save_path}' with accuracy {best_val_acc:.2%}")
        model.load_state_dict(torch.load(save_path))
    
    # Return the trained model, the history of metrics, and the best confusion matrix.
    return model, history, best_cm



def plot_training_history(history, model_name="Custom DenseNet"):
    """Visualizes the training and validation history of a model.

    This function generates and displays two plots: one for training and
    validation loss, and another for validation accuracy. It also highlights
    the epoch where the highest validation accuracy was achieved.

    Args:
        history (dict): A dictionary containing the model's training history.
                        It must include the keys 'val_accuracy', 'val_loss',
                        and 'train_loss'.
        model_name (str, optional): The name of the model, used for plot
                                    titles and labels. Defaults to "Custom DenseNet".
    """
    # Find the index of the epoch with the highest validation accuracy.
    best_epoch_idx = np.argmax(history['val_accuracy'])
    # Get the best validation accuracy and the corresponding validation loss.
    best_val_acc = history['val_accuracy'][best_epoch_idx]
    best_val_loss = history['val_loss'][best_epoch_idx]

    # Print a summary of the model's performance at the best epoch.
    print("---------- Best Epoch Performance ----------")
    print(f"Model: {model_name}")
    print(f"Epoch: {best_epoch_idx + 1}")
    print(f"Validation Accuracy: {best_val_acc:.2%}")
    print(f"Validation Loss:     {best_val_loss:.4f}")
    print("------------------------------------------\n")

    # Set up the figure and subplots for displaying the history.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # Define colors for plot elements to ensure consistency.
    train_color = 'blue'
    val_color = 'red'
    best_epoch_color = 'green'

    # Plot training and validation loss on the first subplot.
    ax1.plot(history['train_loss'], label=f'{model_name} Train Loss', color=train_color, linestyle='-')
    ax1.plot(history['val_loss'], label=f'{model_name} Val Loss', color=val_color, linestyle='--')

    # Highlight the validation loss at the best-accuracy epoch with a marker.
    ax1.plot(best_epoch_idx, best_val_loss, marker='o', color=best_epoch_color, markersize=8, label='Loss When Best Acc Was Achieved')
    # Annotate the marker with its precise value.
    ax1.annotate(f'{best_val_loss:.4f}',
                 xy=(best_epoch_idx, best_val_loss),
                 xytext=(best_epoch_idx, best_val_loss + 0.1),
                 ha='center', color=best_epoch_color,
                 arrowprops=dict(arrowstyle="->", color=best_epoch_color))

    # Set titles and labels for the loss subplot.
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot validation accuracy on the second subplot.
    ax2.plot(history['val_accuracy'], label=f'{model_name} Val Accuracy', color=val_color)

    # Highlight the best validation accuracy with a marker.
    ax2.plot(best_epoch_idx, best_val_acc, marker='o', color=best_epoch_color, markersize=8, label='Best Accuracy Achieved')
    # Annotate the marker with its value.
    ax2.annotate(f'{best_val_acc:.2%}',
                 xy=(best_epoch_idx, best_val_acc),
                 xytext=(best_epoch_idx, best_val_acc - 0.05),
                 ha='center', color=best_epoch_color,
                 arrowprops=dict(arrowstyle="->", color=best_epoch_color))

    # Set titles and labels for the accuracy subplot.
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Determine an appropriate interval for x-axis ticks for readability.
    num_epochs = len(history['train_loss'])
    if num_epochs > 10:
        x_ticks_interval = 2
    else:
        x_ticks_interval = 1

    # Generate tick locations (0-indexed) and corresponding labels (1-indexed).
    tick_locations = np.arange(0, num_epochs, x_ticks_interval)
    tick_labels = np.arange(1, num_epochs + 1, x_ticks_interval)

    # Apply the custom x-axis ticks to both subplots.
    ax1.set_xticks(ticks=tick_locations, labels=tick_labels)
    ax2.set_xticks(ticks=tick_locations, labels=tick_labels)

    # Adjust subplot parameters for a tight layout and display the plot.
    plt.tight_layout()
    plt.show()
    
    
    
def visualize_predictions(model, dataloader, class_names, device):
    """Visualizes model predictions on a sample of images from a dataset.

    This function randomly selects one image from each class in the provided
    dataloader. It then performs inference using the given model and displays
    the images in a grid. Each image is titled with its true and predicted
    labels, colored green for correct predictions and red for incorrect ones.

    Args:
        model (torch.nn.Module): The trained PyTorch model to use for inference.
        dataloader (torch.utils.data.DataLoader): The DataLoader for the dataset to visualize.
        class_names (list of str): A list mapping class indices to their names.
        device (torch.device): The device (e.g., 'cuda', 'cpu') on which to perform inference.
    """
    # Prepare the model for inference.
    model.to(device)
    model.eval()

    # --- Create a mapping from class index to a list of sample indices for that class ---
    # Initialize a dictionary to hold indices for each class.
    class_to_indices = {i: [] for i in range(len(class_names))}
    # Access the targets and indices from the underlying dataset and subset.
    full_dataset_targets = dataloader.dataset.subset.dataset.targets
    subset_indices = dataloader.dataset.subset.indices
    # Populate the dictionary by mapping each sample's true label to its index within the subset.
    for subset_idx, full_idx in enumerate(subset_indices):
        label = full_dataset_targets[full_idx]
        class_to_indices[label].append(subset_idx)
    # ---

    # Create a grid of subplots to display the images.
    fig, axes = plt.subplots(nrows=3, ncols=7, figsize=(18, 8))

    # Disable gradient computations for the inference phase.
    with torch.no_grad():
        # Loop through each class and its corresponding subplot axis.
        for i, ax in enumerate(axes.flatten()):
            # If there are more subplots than classes, turn off the extra ones.
            if i >= len(class_names):
                ax.axis('off')
                continue

            # Randomly select one image index from the current class.
            random_image_idx = random.choice(class_to_indices[i])
            
            # Get the image tensor and its true label from the dataset.
            image_tensor, true_label = dataloader.dataset[random_image_idx]
            
            # Prepare the image tensor for the model by adding a batch dimension and moving it to the device.
            image_batch = image_tensor.unsqueeze(0).to(device)

            # Pass the image through the model to get the output logits.
            outputs = model(image_batch)
            # Determine the predicted class index by finding the index of the maximum logit.
            _, pred = torch.max(outputs, 1)
            predicted_label = pred.item()
            
            # Set the title color to green for correct predictions and red for incorrect ones.
            is_correct = (predicted_label == true_label)
            title_color = 'green' if is_correct else 'red'
            # Set the subplot's title with the predicted and true labels.
            ax.set_title(
                f'Predicted: {class_names[predicted_label]}\n(True: {class_names[true_label]})',
                color=title_color
            )
            
            # Reverse the normalization of the image tensor for proper visualization.
            img_to_plot = unnormalize(image_tensor)
            
            # Convert the tensor to a NumPy array and adjust dimensions for displaying.
            ax.imshow(np.transpose(img_to_plot.numpy(), (1, 2, 0)))
            # Display the image and hide the axis ticks.
            ax.axis('off')

    # Adjust the layout to prevent titles from overlapping and show the plot.
    plt.tight_layout()
    plt.show() 
    
    
    
def plot_confusion_matrix(cm_np, labels):
    """Calculates and displays per-class accuracy, then plots a confusion matrix.

    This function first computes the accuracy for each individual class from the
    provided confusion matrix. It displays these scores with a progress bar, then
    uses scikit-learn's ConfusionMatrixDisplay to visualize the full matrix.

    Args:
        cm_np (numpy.ndarray): The confusion matrix to be plotted, where rows
                               represent true labels and columns represent
                               predicted labels.
        labels (list of str): A list of class names that correspond to the
                              matrix indices.
    """
    # --- Per-Class Accuracy Calculation ---
    correct_predictions = cm_np.diagonal()
    total_samples_per_class = cm_np.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.nan_to_num(correct_predictions / total_samples_per_class)
    class_accuracies = {label: acc for label, acc in zip(labels, per_class_acc)}

    # --- Display Per-Class Accuracy with a Progress Bar ---
    print("--- Per-Class Accuracy ---")
    for class_name, acc in tqdm(class_accuracies.items(), desc="Calculating Metrics"):
        print(f"{class_name:<20} | Accuracy: {acc:.2%}")
        time.sleep(0.05)
    print("-" * 40 + "\n")

    # --- Confusion Matrix Plotting ---
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_np, display_labels=labels)
    
    # Create a figure and axes object with the desired size
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Render the confusion matrix plot on the created axes
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    
    # Rotate the x-axis tick labels for better readability with long names.
    plt.xticks(rotation=45, ha="right")
    
    # Set the plot's title and axis labels.
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # Display the finalized plot.
    plt.show()



def load_pretrained_densenet(num_classes, seed=None, pretrained=True, train_classifier_only=True,
                             weights_path="./pretrained_densenet_weights/densenet121-a639ec97.pth"):
    """Initializes a DenseNet-121 model and prepares it for transfer learning.

    This function loads the DenseNet-121 architecture, optionally with weights
    pre-trained on ImageNet. It then replaces the final classifier layer to
    adapt the model for a new task with a specified number of classes and can
    freeze the feature extraction layers.

    Args:
        num_classes (int): The number of output classes for the new classifier head.
        seed (int, optional): A random seed for reproducible weight initialization
                              of the new classifier. Defaults to None.
        pretrained (bool, optional): If True, loads pre-trained weights from the
                                     specified path. Defaults to True.
        train_classifier_only (bool, optional): If True, freezes the feature extractor
                                                layers so only the new classifier is trained.
                                                Defaults to True.
        weights_path (str, optional): The local file path to the pre-trained weights.

    Returns:
        torch.nn.Module: The configured DenseNet-121 model.
    """
    
    # Set the manual seed for PyTorch for reproducible weight initialization for the classifier head.
    if seed is not None:
        torch.manual_seed(seed)
    
    # Instantiate the DenseNet-121 model from torchvision without its own pretrained weights.
    model = tv_models.densenet121(weights=None)

    # Conditionally load pretrained weights from a local file if the flag is set.
    if pretrained:
        # Check if the specified weights file exists and raise an error if not.
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at path: {weights_path}")

        # Load the state dictionary from the file, mapping it to the CPU to prevent device mismatches.
        state_dict = torch.load(weights_path, map_location='cpu')
        # Apply the loaded weights to the model architecture.
        model.load_state_dict(state_dict)

    # If configured for transfer learning, freeze the parameters of the feature extractor.
    if train_classifier_only:
        # Iterate through all model parameters and disable gradient calculations.
        for param in model.parameters():
            param.requires_grad = False

    # Retrieve the number of input features for the model's original classifier.
    num_ftrs = model.classifier.in_features
    # Replace the original classifier with a new, untrained linear layer.
    # The new layer will have the correct number of output units for the new task.
    # By default, its parameters will have requires_grad=True.
    model.classifier = nn.Linear(num_ftrs, num_classes)

    # Return the modified and configured model.
    return model