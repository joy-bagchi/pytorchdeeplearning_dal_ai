import os
import random
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import HTML, display
from sklearn.metrics import ConfusionMatrixDisplay
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchvision import datasets
from tqdm.auto import tqdm



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
    mean = torch.tensor([0.378, 0.393, 0.345])
    std = torch.tensor([0.205, 0.173, 0.170])
    
    # Create a copy of the tensor to avoid modifying the original in-place.
    unnormalized_tensor = tensor.clone()
    
    # Apply the unnormalization formula to each channel: (pixel * std) + mean.
    for i, (m, s) in enumerate(zip(mean, std)):
        unnormalized_tensor[i].mul_(s).add_(m)
        
    # Clamp pixel values to the valid [0, 1] range to correct for floating-point inaccuracies.
    unnormalized_tensor = torch.clamp(unnormalized_tensor, 0, 1)
    
    # Return the unnormalized tensor.
    return unnormalized_tensor



def display_dataset_stats(base_data_dir):
    """
    Analyzes and displays a statistical summary of an image dataset.

    This function iterates through the subdirectories of a specified base
    directory, where each subdirectory is considered a distinct class. It
    counts the number of image files ('.jpg') within each class and
    renders the statistics in a styled HTML table, including a total count.

    Args:
        base_data_dir (str): The file path to the root directory of the dataset.
    """
    class_counts = {}
    
    # Iterate through the base directory to find class folders and count images.
    try:
        for class_name in os.listdir(base_data_dir):
            class_path = os.path.join(base_data_dir, class_name)
            if os.path.isdir(class_path):
                # Count only files ending with .jpg (case-insensitive)
                image_count = len([
                    f for f in os.listdir(class_path)
                    if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith('.jpg')
                ])
                class_counts[class_name] = image_count
    except FileNotFoundError:
        print(f"Error: The directory '{base_data_dir}' was not found.")
        return

    # Verify that some classes were found before proceeding.
    if not class_counts:
        print(f"No class folders with .jpg images found in '{base_data_dir}'.")
        return

    # Restructure the collected data into a list of dictionaries for the DataFrame.
    data_list = []
    for class_name, count in class_counts.items():
        data_list.append({
            'Class Name': class_name,
            'Number of Images': count
        })

    # Create a DataFrame from the list, sort it by class name, and calculate the total.
    df = pd.DataFrame(data_list).sort_values(by='Class Name').reset_index(drop=True)
    total_images = df['Number of Images'].sum()
    
    # Create a 'Total' summary row and append it to the DataFrame.
    total_row = pd.DataFrame([{
        'Class Name': '<b>Total</b>',
        'Number of Images': total_images
    }])
    df_display = pd.concat([df, total_row], ignore_index=True)

    # Apply CSS styling to the DataFrame for a professional presentation.
    styler = df_display.style.hide(axis="index")
    styler.set_table_styles(
        [
            {"selector": "table", "props": [("width", "60%"), ("margin", "0")]},
            {"selector": "td", "props": [("text-align", "left"), ("padding", "8px")]},
            {"selector": "th", "props": [
                ("text-align", "left"),
                ("padding", "8px"),
                ("background-color", "#4f4f4f"),
                ("color", "white")
            ]}
        ]
    )
    styler.set_properties(**{"white-space": "normal"})
    
    # Render the styled DataFrame.
    display(styler)
    


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



def show_sample_images(dataset):
    """
    Visualizes a random sample image from each class in the dataset.

    This function creates a grid of subplots to display one randomly selected
    image for each available class. It assumes the provided dataset object has
    a `.classes` attribute and is a wrapper around a `torch.utils.data.Subset`,
    which is necessary to efficiently access the original indices and labels.

    Args:
        dataset (Dataset): The dataset to visualize. It must conform to the
                           structure described above.
    """
    # Retrieve the list of class names from the dataset object.
    classes = dataset.classes
    
    # Build a mapping of class indices to the dataset indices for efficient random sampling.
    class_to_indices = {i: [] for i in range(len(classes))}
    full_dataset_targets = dataset.subset.dataset.targets
    subset_indices = dataset.subset.indices
    for subset_idx, full_idx in enumerate(subset_indices):
        label = full_dataset_targets[full_idx]
        class_to_indices[label].append(subset_idx)

    # Create a grid of subplots to display the images.
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(10, 6))
    
    # Iterate over the subplots and populate them with images.
    for i, ax in enumerate(axes.flatten()):
        if i < len(classes):
            class_name = classes[i]
            
            # Select a random image index from the current class.
            random_image_idx = random.choice(class_to_indices[i])
            
            # Retrieve the transformed image and its label from the dataset.
            image, label = dataset[random_image_idx]
            
            # Un-normalize the image for correct color display.
            image = unnormalize(image)
            
            # Convert the tensor to a NumPy array and transpose dimensions for plotting.
            npimg = image.numpy()
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(class_name)

        # Hide the axes for a cleaner look.
        ax.axis('off')
            
    # Adjust subplot layout to prevent titles from overlapping and render the plot.
    plt.tight_layout()
    plt.show()



def display_torch_summary(summary_object, attr_names, display_names, depth):
    """
    Formats and displays a torchinfo summary object as a styled HTML table.

    This utility function processes a summary object generated by the torchinfo
    library, extracts specified layer attributes, and formats them into a
    pandas DataFrame. It then renders the DataFrame as a clean, readable
    HTML table within a Jupyter environment. Key summary statistics, such as
    parameter counts and memory usage, are displayed below the main table.

    Args:
        summary_object: The summary object returned by `torchinfo.summary()`.
        attr_names (list of str): A list of layer attribute names to extract from
                                  the summary (e.g., 'input_size', 'num_params').
        display_names (list of str): A list of desired column headers for the
                                     output table that correspond to attr_names.
        depth (int): The maximum depth of the model hierarchy to display.
    """
    
    # Initialize data structures for building the DataFrame.
    layer_data = []
    display_columns = ["Layer (type:depth-idx)"] + display_names 

    # Iterate through each layer in the summary list.
    for layer in summary_object.summary_list:
        # Skip layers that are deeper than the specified maximum depth.
        if layer.depth > depth:
            continue

        # Initialize a dictionary to hold the data for the current layer's row.
        row = {}
        
        # Construct the hierarchical layer name with appropriate indentation.
        indent = "&nbsp;"*4*layer.depth
        if layer.depth > 0:
            layer_name = f"{layer.class_name}: {layer.depth}-{layer.depth_index}"
        else:
            layer_name = layer.class_name
        
        row["Layer (type:depth-idx)"] = f"{indent}{layer_name}"
        
        # Populate the row dictionary with the specified layer attributes.
        for attr, name in zip(attr_names, display_names):
            # Handle parameter counts separately to show '--' for non-leaf container modules.
            if attr == "num_params":
                show_params = layer.is_leaf_layer or layer.depth == depth
                if show_params and layer.num_params > 0:
                    value = f"{layer.num_params:,}"
                else:
                    value = "--"
            else:
                # Fetch all other attributes directly from the layer object.
                value = getattr(layer, attr, "N/A")
            
            row[name] = value
        layer_data.append(row)

    # Create a pandas DataFrame from the collected layer data.
    df = pd.DataFrame(layer_data, columns=display_columns)
    
    # Apply CSS styling to the DataFrame for a clean HTML presentation.
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
    
    # Convert the styled table to an HTML string.
    table_html = styler.to_html()

    # Compile summary statistics for parameter counts into an HTML block.
    total_params = f"{summary_object.total_params:,}"
    trainable_params = f"{summary_object.trainable_params:,}"
    non_trainable_params = f"{summary_object.total_params - summary_object.trainable_params:,}"
    total_mult_adds = f"{summary_object.total_mult_adds/1e9:.2f} G"
    
    params_html = f"""
    <div style="margin-top: 20px; font-family: monospace; line-height: 1.6;">
        <hr><p><b>Total params:</b> {total_params}</p>
        <p><b>Trainable params:</b> {trainable_params}</p>
        <p><b>Non-trainable params:</b> {non_trainable_params}</p>
        <p><b>Total mult-adds (G):</b> {total_mult_adds}</p><hr>
    </div>"""

    # Compile summary statistics for memory and size estimation.
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

    # Combine the table and summary statistics into a single HTML object and display it.
    final_html = table_html + params_html + size_html
    display(HTML(final_html))


    
def training_loop_16_mixed(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device):
    """
    Executes a complete training and validation loop for a PyTorch model.

    This function handles the full training process. It uses a 16-bit mixed 
    precision training strategy to accelerate performance and reduce memory usage.
    It also tracks does metric tracking (loss, accuracy).

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        loss_function (callable): The loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): The optimization algorithm (e.g., Adam).
        num_epochs (int): The total number of epochs to train for.
        device (torch.device): The device (e.g., 'cuda', 'mps', 'cpu') to run on.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The trained model.
            - history (dict): A dictionary of metrics (loss, accuracy) per epoch.
            - final_cm (numpy.ndarray): The confusion matrix from the final epoch.
    """
    # Determine the device type string for AMP compatibility.
    if device == torch.device("mps"):
        device_str = "mps"
    elif device == torch.device("cuda"):
        device_str = "cuda"
    else:
        device_str = "cpu"

    # Initialize the gradient scaler for AMP, disabled for MPS which does not support it.
    use_scaler = device_str != "mps"
    scaler = GradScaler() if use_scaler else None

    # Move the model to the specified device.
    model.to(device)

    # Initialize a dictionary to store training and validation metrics.
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # Initialize torchmetrics objects for accuracy and confusion matrix calculation.
    num_classes = len(train_loader.dataset.classes)
    val_accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)
    val_cm = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    # --- Main Training Loop ---
    for epoch in range(num_epochs):
        # --- Training Phase ---
        # Set the model to training mode.
        model.train()
        
        # Initialize accumulators for the training phase.
        running_train_loss = 0.0
        train_samples_processed = 0
        
        # Create a progress bar for the training loader.
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False
        )

        for inputs, labels in train_pbar:
            # Move data to the specified device.
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Clear the gradients from the previous iteration.
            optimizer.zero_grad(set_to_none=True)

            # Use automatic mixed precision for the forward pass to improve performance.
            with autocast(device_type=device_str, dtype=torch.float16):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            # Perform backpropagation with the gradient scaler if enabled.
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            # Perform standard backpropagation if the scaler is not used (e.g., on MPS).
            else:
                loss.backward()
                optimizer.step()

            # Update and display the running loss on the progress bar.
            batch_size = inputs.size(0)
            running_train_loss += loss.item() * batch_size
            train_samples_processed += batch_size
            display_loss = running_train_loss / train_samples_processed
            train_pbar.set_postfix(loss=f"{display_loss:.4f}")

        # Calculate and store the average training loss for the epoch.
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)

        # --- Validation Phase ---
        # Set the model to evaluation mode.
        model.eval()
        
        # Reset accumulators and metrics for the validation phase.
        running_val_loss = 0.0
        val_samples_processed = 0
        val_accuracy.reset()
        val_cm.reset()

        # Create a progress bar for the validation loader.
        val_pbar = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False
        )
        # Disable gradient calculations for the validation phase.
        with torch.no_grad():
            for inputs, labels in val_pbar:
                # Move data to the specified device.
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Use automatic mixed precision for the validation forward pass.
                with autocast(device_type=device_str, dtype=torch.float16):
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)

                # Update validation metrics with the results from the current batch.
                preds = outputs.argmax(dim=1)
                batch_size = inputs.size(0)
                running_val_loss += loss.item() * batch_size
                val_samples_processed += batch_size
                val_accuracy.update(preds, labels)
                val_cm.update(preds, labels)

                # Update the progress bar with current validation loss and accuracy.
                current_acc = val_accuracy.compute().item()
                display_loss = running_val_loss / val_samples_processed
                val_pbar.set_postfix(
                    acc=f"{current_acc:.2%}", loss=f"{display_loss:.4f}"
                )

        # Calculate and store the average validation loss and accuracy for the epoch.
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = val_accuracy.compute().item()
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)

        # Print a summary of the epoch's performance.
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Val Loss: {epoch_val_loss:.4f}, "
            f"Val Acc: {epoch_val_acc:.4f}"
        )

    # Compute the final confusion matrix after all epochs are complete.
    final_cm = val_cm.compute().cpu().numpy()
    
    # Return the trained model, metrics history, and confusion matrix.
    return model, history, final_cm



def plot_training_logs(history1, history2, model_name1="PlainCNN Model", model_name2="SimpleResNet Model"):
    """
    Plots and compares the training history of two models.

    Args:
        history1 (dict): The training history dictionary for the first model.
        history2 (dict): The training history dictionary for the second model.
        model_name1 (str, optional): The name of the first model for labels.
                                     Defaults to "Plain CNN Model".
        model_name2 (str, optional): The name of the second model for labels.
                                     Defaults to "ResNet Model".
    """
    # Extract the final validation accuracy for each model from the history.
    final_acc1 = history1['val_accuracy'][-1]
    final_acc2 = history2['val_accuracy'][-1]

    # Display a summary of the final validation metrics.
    print("---------- Final Validation Accuracies ---------")
    print(f"{model_name1:<15}     |  Accuracy: {final_acc1:.2%}")
    print(f"{model_name2:<15}  |  Accuracy: {final_acc2:.2%}")
    print("------------------------------------------------\n")

    # Create a figure with two side-by-side subplots for loss and accuracy.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    color1, color2 = 'red', 'blue'

    # Plot the training and validation loss curves for both models.
    ax1.plot(history1['train_loss'], label=f'{model_name1} Train Loss', color=color1, linestyle='-')
    ax1.plot(history1['val_loss'], label=f'{model_name1} Val Loss', color=color1, linestyle='--')
    ax1.plot(history2['train_loss'], label=f'{model_name2} Train Loss', color=color2, linestyle='-')
    ax1.plot(history2['val_loss'], label=f'{model_name2} Val Loss', color=color2, linestyle='--')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot the validation accuracy curves for both models.
    ax2.plot(history1['val_accuracy'], label=f'{model_name1} Val Accuracy', color=color1)
    ax2.plot(history2['val_accuracy'], label=f'{model_name2} Val Accuracy', color=color2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Configure the x-axis ticks to be displayed at a dynamic interval.
    num_epochs = len(history1['train_loss'])
    if num_epochs > 10:
        x_ticks_interval = 2
    else:
        x_ticks_interval = 1
    
    # Define the locations for the ticks (0-indexed) and their labels (1-indexed).
    tick_locations = np.arange(0, num_epochs, x_ticks_interval)
    tick_labels = np.arange(1, num_epochs + 1, x_ticks_interval)

    # Apply the custom ticks and labels to both subplots.
    ax1.set_xticks(ticks=tick_locations, labels=tick_labels)
    ax2.set_xticks(ticks=tick_locations, labels=tick_labels)
    
    # Adjust layout to prevent titles from overlapping and render the plot.
    plt.tight_layout()
    plt.show()


    
def visualize_predictions(model, dataloader, classes, device):
    """
    Displays a grid of predictions for one random image from each class.

    This function sets the model to evaluation mode, intelligently samples one
    random image from each unique class in the dataset, and performs inference.
    It then displays the results in a 3x5 grid, annotating each image with
    its true and predicted label.

    Args:
        model (nn.Module): The trained PyTorch model for inference.
        dataloader (DataLoader): DataLoader for the validation set.
        classes (list of str): A list of all class names for displaying labels.
        device (torch.device): The device (e.g., 'cuda', 'cpu') to run inference on.
    """
    # Move the model to the specified device and set it to evaluation mode.
    model.to(device)
    model.eval()

    # --- Find one random image index for each class ---
    # This creates a map of {class_index: [list_of_sample_indices]}
    class_to_indices = {i: [] for i in range(len(classes))}
    # Access the underlying dataset to get all labels and indices
    full_dataset_targets = dataloader.dataset.subset.dataset.targets
    subset_indices = dataloader.dataset.subset.indices
    for subset_idx, full_idx in enumerate(subset_indices):
        label = full_dataset_targets[full_idx]
        class_to_indices[label].append(subset_idx)
    # ---

    # Create a 3x5 grid of subplots for the 15 classes.
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 9))

    # Disable gradient calculations for inference.
    with torch.no_grad():
        # Iterate through each class and its corresponding subplot axis.
        for i, ax in enumerate(axes.flatten()):
            # Ensure we don't try to access a class that doesn't exist.
            if i >= len(classes):
                ax.axis('off')
                continue

            # Select a random image from the current class's list of indices.
            random_image_idx = random.choice(class_to_indices[i])
            
            # Retrieve the transformed image and its true label from the dataset.
            image_tensor, true_label = dataloader.dataset[random_image_idx]
            
            # Add a batch dimension and move the image to the specified device for the model.
            image_batch = image_tensor.unsqueeze(0).to(device)

            # Perform inference to get the model's prediction.
            outputs = model(image_batch)
            _, pred = torch.max(outputs, 1)
            predicted_label = pred.item()
            
            # Determine if the prediction was correct for title coloring.
            is_correct = (predicted_label == true_label)
            title_color = 'green' if is_correct else 'red'
            ax.set_title(
                f'Predicted: {classes[predicted_label]}\n(True: {classes[true_label]})',
                color=title_color
            )
            
            # Un-normalize the image tensor for correct color display.
            img_to_plot = unnormalize(image_tensor)
            
            # Convert tensor to numpy array and transpose for plotting.
            ax.imshow(np.transpose(img_to_plot.numpy(), (1, 2, 0)))
            ax.axis('off')

    # Adjust layout to prevent titles from overlapping and render the plot.
    plt.tight_layout()
    plt.show()
    
    
    
def plot_confusion_matrix(cm_np, labels):
    """
    Calculates and displays per-class accuracy, then plots a confusion matrix.

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
    # The diagonal contains the correct predictions for each class.
    correct_predictions = cm_np.diagonal()
    
    # The sum of each row is the total number of actual samples for that class.
    total_samples_per_class = cm_np.sum(axis=1)
    
    # Calculate accuracy, handling division-by-zero for classes with no samples.
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.nan_to_num(correct_predictions / total_samples_per_class)
    
    # Create a dictionary mapping class labels to their accuracy.
    class_accuracies = {label: acc for label, acc in zip(labels, per_class_acc)}

    # --- Display Per-Class Accuracy with a Progress Bar ---
    print("--- Per-Class Accuracy ---")
    # Use tqdm to create a progress bar while iterating through and printing results.
    for class_name, acc in tqdm(class_accuracies.items(), desc="Calculating Metrics"):
        print(f"{class_name:<20} | Accuracy: {acc:.2%}")
        time.sleep(0.05) # Pause briefly to make the progress bar visible
    print("-" * 40 + "\n")

    # --- Confusion Matrix Plotting ---
    # Create a confusion matrix display object from the matrix and labels.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_np, display_labels=labels)

    # Render the confusion matrix plot with a blue color map.
    disp.plot(cmap=plt.cm.Blues)
    
    # Rotate the x-axis tick labels for better readability with long names.
    plt.xticks(rotation=45, ha="right")
    
    # Set the plot's title and axis labels.
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # Display the finalized plot.
    plt.show()