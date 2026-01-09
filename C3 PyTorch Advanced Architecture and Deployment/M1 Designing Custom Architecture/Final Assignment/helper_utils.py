import copy
import io
import os
import random
from collections import Counter, defaultdict

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display, HTML
from PIL import Image, ExifTags
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets
from tqdm.auto import tqdm



def load_datasets(dataset_path, train_transform=None, val_transform=None):
    """
    Loads training and validation datasets from specified subdirectories.

    This function uses torchvision's ImageFolder to load datasets from
    'sub_train_dataset' and 'sub_validation_dataset' folders within
    the provided root path.

    Args:
        dataset_path: The root directory path containing the dataset subfolders.
        train_transform: An optional torchvision transform to be applied
                         to the training dataset.
        val_transform: An optional torchvision transform to be applied
                       to the validation dataset.

    Returns:
        A tuple containing the training dataset and the validation dataset.
    """
    # Load the training dataset from the 'sub_train_dataset' subdirectory
    train_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, "sub_train_dataset"), transform=train_transform
    )
    # Load the val dataset from the 'sub_validation_dataset' subdirectory
    validation_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, "sub_validation_dataset"), transform=val_transform
    )
    
    # Return the loaded datasets
    return train_dataset, validation_dataset

    
    
    
def create_dataloaders(train_dataset, validation_dataset, batch_size):
    """
    Creates and returns DataLoader instances for training and validation.

    Args:
        train_dataset: The dataset to be used for training.
        validation_dataset: The dataset to be used for validation.
        batch_size: The number of samples per batch to load.

    Returns:
        A tuple containing the training DataLoader and the validation DataLoader.
    """
    # Create a DataLoader for the training dataset, with shuffling enabled
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Create a DataLoader for the validation dataset, without shuffling
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    # Return the created DataLoaders
    return train_loader, val_loader
    
    
    
def unnormalize(tensor_img, mean=[0.5] * 3, std=[0.5] * 3):
    """Reverses the normalization on a tensor image."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor_img * std + mean


def show_sample_images(dataset):
    """
    Visualizes a random sample image from each class in the dataset.

    This function creates a grid of subplots to display one randomly selected
    image for each class. It expects the dataset to have a `.classes` 
    attribute and a `.targets` attribute (like an ImageFolder).

    Args:
        dataset: The dataset to visualize (e.g., an ImageFolder).
    """
    # Retrieve the list of class names from the dataset object
    classes = dataset.classes
    
    # Build a mapping of class labels to their corresponding indices within the subset
    # Initialize a dictionary to hold lists of indices for each class
    class_to_indices = {i: [] for i in range(len(classes))}

    # Access the targets directly from the ImageFolder dataset
    all_targets = dataset.targets
    # Iterate over all targets to build the map
    for index, label in enumerate(all_targets):
        # Store the dataset index in the list for its corresponding label
        class_to_indices[label].append(index)

    # Create a grid of subplots (1 rows, 7 columns) to display the images
    fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(10, 6))
    
    # Iterate over the subplots and populate them with images
    for i, ax in enumerate(axes.flatten()):
        # Only proceed if the current index corresponds to a valid class
        if i < len(classes):
            # Get the class name for the current index
            class_name = classes[i]
            
            # Select a random image index from the list for the current class
            random_image_idx = random.choice(class_to_indices[i])
            
            # Retrieve the transformed image and its label using the subset index
            image, label = dataset[random_image_idx]
            
            # Un-normalize the image tensor for correct color display
            image = unnormalize(image)
            
            # Convert the tensor to a NumPy array
            npimg = image.numpy()
            # Transpose dimensions from (C, H, W) to (H, W, C) for plotting
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            # Set the title of the subplot to the class name
            ax.set_title(class_name)

        # Hide the axes (ticks and labels) for a cleaner look
        ax.axis('off')
            
    # Adjust subplot layout to prevent titles from overlapping
    plt.tight_layout()
    # Render the plot
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
    
    
    
def compute_class_weights(dataset):
    """
    Computes class weights inversely proportional to class frequencies.

    Args:
        dataset (torch.utils.data.Dataset): A PyTorch dataset, expected to have
                                             a 'targets' attribute (like torchvision ImageFolder)
                                             or provide labels when iterated.

    Returns:
        torch.Tensor: A tensor containing the weight for each class.
    """
    # Try to get labels directly if the dataset supports it (like ImageFolder)
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'labels'): # Handle datasets with a 'labels' attribute
         labels = dataset.labels
    else:
        # Fallback: Iterate through the dataset to get labels (slower)
        print("Dataset has no 'targets' or 'labels' attribute, iterating to get labels...")
        labels = [label for _, label in dataset]

    # Count the frequency of each class label
    class_counts = Counter(labels)

    # Sort counts by class index (0, 1, 2, ...) to ensure correct weight order
    sorted_counts = [class_counts[i] for i in sorted(class_counts)]

    # Calculate weights: total_samples / (num_classes * count_for_this_class)
    # This is a common formula for inverse frequency weighting
    total_samples = len(labels)
    num_classes = len(sorted_counts)

    # Calculate weight for each class
    weights = []
    for count in sorted_counts:
        weight = total_samples / (num_classes * count)
        weights.append(weight)

    # Convert weights list to a PyTorch tensor
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    return weights_tensor    
    
    
    
def training_loop(model, train_loader, validation_loader, loss_fcn, optimizer, scheduler, device, n_epochs):
    """
    Executes a training and validation loop, returning the best model.

    This function iterates for a specified number of epochs, running one
    training phase and one validation phase per epoch. It tracks the best
    validation accuracy and returns the model state from that epoch.

    Args:
        model: The neural network model to be trained.
        train_loader: DataLoader for the training dataset.
        validation_loader: DataLoader for the validation dataset.
        loss_fcn: The loss function to use for training.
        optimizer: The optimization algorithm.
        scheduler: An optional learning rate scheduler.
        device: The device (e.g., 'cuda' or 'cpu') to run the training on.
        n_epochs: The total number of epochs to train.
    
    Returns:
        The trained model (nn.Module) with the weights from the epoch
        that achieved the highest validation accuracy.
    """
    
    # --- Setup ---
    # Move model to the target device
    model.to(device)
    
    # Get num_classes from the dataset (common in ImageFolder)
    num_classes = len(train_loader.dataset.classes)
    # Initialize metric calculators
    train_acc_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    val_acc_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    
    # --- Best Model Tracking ---
    # Initialize the best validation accuracy found so far
    best_val_acc = 0.0
    # Initialize a tracker for the epoch number of the best model
    best_epoch = 0
    # Variable to store the state dictionary of the best model
    best_model_state = None
    # --- End Tracking ---
    
    print("--- Starting Training ---")
    
    # --- Main Epoch Loop ---
    # Iterate over the specified number of epochs
    for epoch in range(n_epochs):
        
        # --- Training Phase ---
        # Set the model to training mode
        model.train()
        # Initialize running loss for the current epoch
        running_train_loss = 0.0
        # Reset the training accuracy metric at the start of each epoch
        train_acc_metric.reset()
        
        # Create a TQDM progress bar for the training batches
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [TRAIN]", leave=False)
        
        # Iterate over batches of training data
        for inputs, labels in train_pbar:
            # Move data to the device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 1. Zero gradients
            # Clear previously calculated gradients
            optimizer.zero_grad()
            
            # 2. Forward pass
            # Get model predictions for the current batch
            outputs = model(inputs)
            
            # 3. Calculate loss
            # Compute the loss between predictions and actual labels
            loss = loss_fcn(outputs, labels)
            
            # 4. Backward pass
            # Compute gradients of the loss with respect to model parameters
            loss.backward()
            
            # 5. Optimizer step
            # Update model parameters based on the computed gradients
            optimizer.step()
            
            # --- Update Metrics ---
            # Accumulate the training loss
            running_train_loss += loss.item() * inputs.size(0)
            # Get the predicted class indices
            preds = torch.argmax(outputs, dim=1)
            # Update the accuracy metric
            train_acc_metric.update(preds, labels)
            
            # Update progress bar postfix
            train_pbar.set_postfix(batch_loss=loss.item())

        # Calculate average training loss and accuracy for the epoch
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = train_acc_metric.compute().item()
        
        
        # --- Validation Phase ---
        # Set the model to evaluation mode
        model.eval()
        # Initialize running validation loss
        running_val_loss = 0.0
        # Reset the validation accuracy metric
        val_acc_metric.reset()
        
        # Create a TQDM progress bar for the validation batches
        val_pbar = tqdm(validation_loader, desc=f"Epoch {epoch+1}/{n_epochs} [VAL]", leave=False)
        
        # Disable gradient calculation for validation
        with torch.no_grad():
            # Iterate over batches of validation data
            for inputs, labels in val_pbar:
                # Move data to the device
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 1. Forward pass
                # Get model predictions
                outputs = model(inputs)
                
                # 2. Calculate loss
                # Compute the validation loss
                loss = loss_fcn(outputs, labels)
                
                # --- Update Metrics ---
                # Accumulate the validation loss
                running_val_loss += loss.item() * inputs.size(0)
                # Get the predicted class indices
                preds = torch.argmax(outputs, dim=1)
                # Update the validation accuracy metric
                val_acc_metric.update(preds, labels)
                
                # Update progress bar postfix
                val_pbar.set_postfix(batch_loss=loss.item())

        # Calculate average validation loss and accuracy for the epoch
        epoch_val_loss = running_val_loss / len(validation_loader.dataset)
        epoch_val_acc = val_acc_metric.compute().item()
        

        # --- End of Epoch ---
        
        # Step the scheduler
        if scheduler is not None:
            # Check if it's ReduceLROnPlateau, which needs a metric
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                # Step the scheduler based on validation loss
                scheduler.step(epoch_val_loss)
            else:
                # Step other types of schedulers
                scheduler.step()
        
        # Log metrics for the epoch
        print(f"Epoch {epoch+1}/{n_epochs} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        # --- Check for Best Model ---
        # If current validation accuracy is better than the best seen so far
        if epoch_val_acc > best_val_acc:
            # Update the best validation accuracy
            best_val_acc = epoch_val_acc
            # Store the current epoch number
            best_epoch = epoch + 1
            # Use copy.deepcopy to save a snapshot of the model's state
            best_model_state = copy.deepcopy(model.state_dict())
            # Print a message indicating a new best model
            print(f"    ^ New best model found!")
        # --- End Check ---

    print("\n--- Training Complete ---")
    
    # --- Load Best Model ---
    # Check if a best model state was saved
    if best_model_state is not None:
        print(f"\nReturning best model from epoch {best_epoch} with {best_val_acc:.4f} validation accuracy.")
        # Load the best performing weights back into the model
        model.load_state_dict(best_model_state)
    else:
        # Warn if no improvement was seen and the last model is being returned
        print("\nWarning: No best model state was saved (e.g., validation never improved). Returning last model.")
    
    # Return the best model
    return model



def display_random_predictions_per_class(model, val_loader, classes, device, num_classes=7):
    """
    Displays one sample image for each class from the validation data.

    Args:
        model: The trained model to use for predictions.
        val_loader: DataLoader containing the validation dataset.
        classes: A list of class names corresponding to the model's output.
        device: The device (e.g., 'cuda' or 'cpu') to run inference on.
        num_classes: The total number of classes to look for.
    """
    # Set the model to evaluation mode (e.g., disable dropout)
    model.eval()
    # Move the model to the specified computation device
    model.to(device)

    # Dictionary to store one image tensor for each class index
    class_images = defaultdict(list)
    # Set to keep track of which class indices have been found
    found_classes = set()

    # Iterate through the validation loader to find one image per class
    # Disable gradient calculations during inference
    with torch.no_grad():
        # Loop over batches of images and labels
        for images, labels in val_loader:
            # Move data to the specified device
            images, labels = images.to(device), labels.to(device)

            # Check each image in the batch
            for i in range(images.size(0)):
                # Get the integer label for the current image
                label_idx = labels[i].item()
                # Check if this class is needed and hasn't been found yet
                if label_idx not in found_classes and label_idx < num_classes :
                    # Store the image tensor (move to CPU for storage/plotting)
                    class_images[label_idx].append(images[i].cpu())
                    # Mark this class index as found
                    found_classes.add(label_idx)

            # If an image for every class has been found, stop iterating
            if len(found_classes) == num_classes:
                # Exit the loop
                break
        # This 'else' block runs if the 'for' loop completes without breaking
        else:
            # Print a warning if not all classes were found in the dataset
            print(f"Warning: Could only find images for {len(found_classes)} out of {num_classes} classes.")


    # --- Prepare images and predictions for display ---
    # List to hold the image tensors for plotting
    selected_images = []
    # List to hold the true class names
    true_labels = []
    # List to hold the predicted class names
    pred_labels = []

    # Get a sorted list of the class indices that were actually found
    sorted_class_indices = sorted([idx for idx in class_images.keys() if idx < num_classes])

    # Perform inference on the selected images
    with torch.no_grad():
        # Loop through the found class indices in order
        for class_idx in sorted_class_indices:
            # Get the first image stored for this class
            img_tensor = class_images[class_idx][0]
            # Add it to the list for plotting
            selected_images.append(img_tensor)

            # Prepare the image for the model (add batch dimension and move to device)
            img_tensor_batch = img_tensor.unsqueeze(0).to(device)
            # Get the model's raw output (logits)
            output = model(img_tensor_batch)
            # Find the index with the highest score (the predicted class)
            _, predicted_idx = torch.max(output, 1)

            # Store the corresponding class names for true and predicted labels
            true_labels.append(classes[class_idx])
            pred_labels.append(classes[predicted_idx.item()])

    # --- Display the images and labels using subplots ---
    # Check if any images were actually selected
    if not selected_images:
        # Inform user if no images are available
        print("No images were selected to display.")
        # Exit the function
        return

    # Get the number of images that will be plotted
    num_found = len(selected_images)
    # Create a figure and a set of subplots (one for each image)
    fig, axes = plt.subplots(1, num_found, figsize=(3 * num_found, 4.5))

    # Handle the case where only one image is found (axes is not an array)
    if num_found == 1:
        # Wrap the single axis in a list to allow iteration
        axes = [axes]

    # Iterate through the selected images and their labels
    for i in range(num_found):
        # Get the specific axis for this plot
        ax = axes[i]
        # Denormalize the image (assuming normalization was mean=0.5, std=0.5)
        img = selected_images[i] * 0.5 + 0.5
        # Convert tensor from (Channel, Height, Width) to (Height, Width, Channel) for plotting
        img = np.transpose(img.numpy(), (1, 2, 0))
        # Display the image
        ax.imshow(img)

        # Determine the color for the prediction text (green for correct, red for incorrect)
        pred_color = 'green' if true_labels[i] == pred_labels[i] else 'red'

        # Set the title to show the true and predicted labels, coloring the prediction
        ax.set_title(f"True: {true_labels[i]}\nPred: {pred_labels[i]}", fontsize=13, color=pred_color)

        # Hide the x and y axes
        ax.axis('off')

    # Adjust the layout to prevent titles and images from overlapping
    plt.tight_layout(pad=1.5)
    # Show the plot
    plt.show()

    

def siamese_training_loop(model, dataloader, loss_fcn, optimizer, device, n_epochs=5):
    """
    Executes the training process for a Siamese network model **in-place**.

    This function iterates over the provided dataloader for a specified
    number of epochs, computes the loss, updates the model weights, and ensures
    the model's final state corresponds to the epoch with the lowest average loss.

    Args:
        model (nn.Module): The neural network model to be trained. **It will be modified in-place.**
        dataloader (DataLoader): The DataLoader providing the training data (e.g., triplets).
        loss_fcn (callable): The loss function used for training (e.g., TripletMarginLoss).
        optimizer (torch.optim.Optimizer): The optimization algorithm (e.g., Adam).
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to perform training on.
        n_epochs (int): The total number of training epochs.

    Returns:
        None: The model is trained in-place and its state is set to the best epoch.
    """
    # Move the model to the target device (CPU or GPU)
    model.to(device)

    # Variables to track the best performing epoch
    best_loss = float('inf') # Initialize best loss to infinity
    best_epoch = 0
    best_model_state = None # To store the state dict of the best model

    # Print a message to indicate the start of training
    print("\n--- Starting Siamese Training ---")

    # Loop over the dataset for the specified number of epochs
    for epoch in range(1, n_epochs + 1):
        # Set the model to training mode (enables dropout, batch norm updates, etc.)
        model.train()
        # Initialize a running loss counter for the current epoch
        running_loss = 0.0
        # Wrap the dataloader with tqdm for a visual progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{n_epochs}", leave=False)

        # Iterate over batches of data from the dataloader
        for data_batch in progress_bar:
            # Unpack the data batch (anchor, positive, negative samples)
            anchor, positive, negative = data_batch
            # Move all tensors in the batch to the selected device
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )

            # --- Training Step ---
            # 1. Clear the gradients from the previous iteration
            optimizer.zero_grad()

            # 2. Perform a forward pass to get the model's outputs (embeddings)
            anchor_out, positive_out, negative_out = model(anchor, positive, negative)

            # 3. Calculate the loss based on the model's outputs
            loss = loss_fcn(anchor_out, positive_out, negative_out)

            # 4. Perform a backward pass to compute gradients
            loss.backward()
            # 5. Update the model's weights using the optimizer
            optimizer.step()
            # --- End Training Step ---

            # Add the loss from the current batch to the running total
            running_loss += loss.item()
            # Update the progress bar's postfix to show the current average loss
            # Use progress_bar.n + 1 for correct average calculation during iteration
            progress_bar.set_postfix(loss=f"{running_loss / (progress_bar.n + 1):.4f}")

        # Calculate the average loss for the entire epoch
        epoch_loss = running_loss / len(dataloader)
        # Print the average loss for the completed epoch
        print(f"Epoch {epoch}/{n_epochs} finished, Average Loss: {epoch_loss:.4f}")

        # --- Check if this epoch is the best so far ---
        if epoch_loss < best_loss:
            best_loss = epoch_loss # Update best loss
            best_epoch = epoch # Update best epoch number
            # Save the model's state dictionary on the CPU
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"    -> New best model saved (Epoch {best_epoch}, Loss: {best_loss:.4f})")
        # --- End Check ---

    # --- Training Complete ---
    print("\n--- Siamese Training Complete ---")

    # Load the best model weights found during training
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model weights from Epoch {best_epoch} with lowest loss: {best_loss:.4f}")
    else:
        print("Warning: No best model state was saved (check training loop). Using weights from the last epoch.")
    # No return statement needed, model was trained in-place and set to best state



def upload_jpg_widget():
    """
    Creates and displays a file upload widget for JPG images.

    This function facilitates the uploading of JPG files within a Jupyter
    or IPython environment. It handles file validation for format and size,
    corrects image orientation based on EXIF data, resizes large images
    while preserving aspect ratio, and saves the final image to a local
    directory.
    """
    # Define the directory where uploaded images will be stored.
    output_image_folder = "./images"
    # Create the destination directory if it does not already exist.
    os.makedirs(output_image_folder, exist_ok=True)

    # Initialize the file upload widget, accepting only JPG/JPEG files.
    uploader = widgets.FileUpload(
        accept='.jpg,.jpeg',
        multiple=False,
        description='Upload JPG (Max 5MB)'
    )

    # Create an output widget to display messages to the user.
    output_area = widgets.Output()

    def on_file_uploaded(change):
        """
        Processes the uploaded file upon a change event.

        This callback function is triggered when a file is uploaded. It validates
        the file's format and size, corrects orientation, resizes if necessary,
        and saves the processed image.

        Args:
            change: A dictionary-like object containing information about the change event.
                    The new file data is in `change['new']`.
        """
        # Retrieve the tuple of uploaded file data from the change event.
        current_uploaded_value_tuple = change['new']
        # If the tuple is empty, it means the upload was cleared, so do nothing.
        if not current_uploaded_value_tuple:
            return

        # Use the output area to display feedback.
        with output_area:
            # Clear any previous messages.
            output_area.clear_output()

            # Get the dictionary containing file data from the tuple.
            file_data_dict = current_uploaded_value_tuple[0]
            # Extract the filename and its binary content.
            filename = file_data_dict['name']
            file_content = file_data_dict['content']

            # Validate that the file has a '.jpg' or '.jpeg' extension.
            if not filename.lower().endswith(('.jpg', '.jpeg')):
                # Format an error message for invalid file types.
                error_msg_format = (
                    f"<p style='color:red;'>Error: Please upload a file with a ‘.jpg’ or ‘.jpeg’ format. "
                    f"You uploaded: '{filename}'</p>"
                )
                # Display the error message and reset the uploader.
                display(HTML(error_msg_format))
                uploader.value = ()
                return

            # Check if the file size exceeds the 5MB limit.
            if len(file_content) > 5 * 1024 * 1024:
                # Calculate file size in megabytes for the error message.
                file_size_mb = len(file_content) / (1024 * 1024)
                # Format an error message for oversized files.
                error_msg_size = (
                    f"<p style='color:red;'>Error: File '{filename}' is too large ({file_size_mb:.2f} MB). "
                    f"Please upload a file less than or equal to 5 MB.</p>"
                )
                # Display the error message and reset the uploader.
                display(HTML(error_msg_size))
                uploader.value = ()
                return

            try:
                # Open the image from its binary content.
                img = Image.open(io.BytesIO(file_content))

                # Attempt to correct image orientation using EXIF data.
                try:
                    # Map EXIF tags to their names for easier lookup.
                    orientation_map = {
                        ExifTags.TAGS[k]: k for k in ExifTags.TAGS if k in ExifTags.TAGS
                    }
                    # Retrieve the EXIF data from the image.
                    exif = img._getexif()

                    # Check if orientation data exists in the EXIF information.
                    if exif and orientation_map['Orientation'] in exif:
                        orientation = exif[orientation_map['Orientation']]
                        # Apply rotation based on the orientation value.
                        if orientation == 3:
                            img = img.transpose(Image.ROTATE_180)
                        elif orientation == 6:
                            img = img.transpose(Image.ROTATE_270)
                        elif orientation == 8:
                            img = img.transpose(Image.ROTATE_90)
                # Handle cases where EXIF data is missing or corrupt.
                except (AttributeError, KeyError, IndexError):
                    pass

                # Get the dimensions of the potentially reoriented image.
                width, height = img.size
                # Check if the image dimensions exceed the 1000x1000 pixel limit.
                if width > 1000 or height > 1000:
                    # Calculate the scaling factor to maintain the aspect ratio.
                    scaling_factor = 1000 / max(width, height)
                    # Compute the new dimensions.
                    new_width = int(width * scaling_factor)
                    new_height = int(height * scaling_factor)
                    # Resize the image using a high-quality downsampling filter.
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Create an in-memory byte stream to save the processed image.
                output_byte_stream = io.BytesIO()
                # Save the image to the byte stream in JPEG format.
                img.save(output_byte_stream, format='JPEG', quality=90)
                # Get the binary content from the stream.
                content_to_write = output_byte_stream.getvalue()

                # Define the full path to save the file.
                save_path = os.path.join(output_image_folder, filename)
                # Write the final image content to a file on disk.
                with open(save_path, 'wb') as f:
                    f.write(content_to_write)

                # Create a string representation of the path for user-friendly output.
                python_code_path = repr(save_path)
                # Format a success message for the user.
                success_message = f"""
                <p style='color:green;'>File successfully uploaded!</p>
                <p>Please use the path as <code>image_path = {python_code_path}</code></p>
                """
                # Display the success message.
                display(HTML(success_message))

            # Catch any exceptions that occur during image processing.
            except Exception as e:
                # Format and display a generic error message.
                error_msg_save = f"<p style='color:red;'>Error processing file '{filename}': {e}</p>"
                display(HTML(error_msg_save))
            # The finally block ensures the uploader is cleared regardless of success or failure.
            finally:
                uploader.value = ()

    # Register the callback function to be executed when a file is uploaded.
    uploader.observe(on_file_uploaded, names='value')
    # Display the file upload widget.
    display(uploader)
    # Display the area for output messages.
    display(output_area)



def get_query_img(image_path):
    """
    Loads and converts an image file to RGB format.

    Args:
        image_path: The file path to the image.

    Returns:
        The loaded and converted image object.
    """
    # Open the image file from the specified path and convert it to the RGB color space
    img = Image.open(image_path).convert("RGB")
    # Return the processed image
    return img



def get_embeddings(model_representation, labeled_dataset, device):
    """
    Computes and returns the feature embeddings for a given dataset using a model.

    Args:
        model_representation: The feature extraction model.
        labeled_dataset: The dataset containing the data samples.
        device: The computational device (e.g., 'cpu' or 'cuda').

    Returns:
        A NumPy array containing the computed embeddings for the entire dataset.
    """
    # Create a DataLoader to process the dataset in batches
    dataloader = DataLoader(labeled_dataset, batch_size=32, shuffle=False)
    # Set the model to evaluation mode (e.g., disables dropout)
    model_representation.eval()
    # Initialize an empty list to store embeddings from each batch
    embeddings = []

    # Disable gradient computation during this block for efficiency
    with torch.no_grad():
        # Iterate over all batches in the dataloader
        for img, _ in dataloader:
            # Move the batch of images to the specified device
            img = img.to(device)
            # Perform a forward pass to get the embeddings
            embedding_img = model_representation.forward(img)
            # Move embeddings to CPU, convert to NumPy, and append to the list
            embeddings.append(embedding_img.cpu().numpy())

    # Concatenate the list of batch embeddings into a single NumPy array
    embeddings = np.concatenate(embeddings, axis=0)
    # Return the complete array of embeddings
    return embeddings



def find_closest(embeddings, target_embedding, num_samples=5):
    """
    Finds the indices of the samples with the closest embeddings to the target embedding.

    Args:
        embeddings: A NumPy array containing the feature embeddings of all samples.
        target_embedding: A NumPy array representing the embedding of the query sample.
        num_samples: The number of closest sample indices to retrieve.

    Returns:
        A NumPy array of the indices corresponding to the closest samples.
    """
    # Calculate Euclidean distances from the target to all other embeddings
    distances = np.linalg.norm(embeddings - target_embedding, axis=1)

    # Get indices of the closest samples by sorting the distances
    # We slice from 1 to num_samples + 1 to exclude the target itself (which has distance 0)
    closest_indices = np.argsort(distances)[
        1 : num_samples + 1
    ]

    # Return the indices of the closest samples
    return closest_indices



def get_image(dataset, index):
    """
    Retrieves an original image and its label from a dataset by index.

    Args:
        dataset: The dataset object (e.g., a torchvision ImageFolder)
                 that contains a 'samples' list of (path, label) tuples.
        index: The index of the item to retrieve.

    Returns:
        A tuple containing the PIL Image object (in RGB format) and
        its corresponding label.
    """
    # Get the file path and label for the given index from the dataset's samples
    path, label = dataset.samples[index]  
    # Open the image file using its path and convert it to RGB format
    original_img = Image.open(path).convert("RGB")
    # Return the loaded image and its label
    return original_img, label