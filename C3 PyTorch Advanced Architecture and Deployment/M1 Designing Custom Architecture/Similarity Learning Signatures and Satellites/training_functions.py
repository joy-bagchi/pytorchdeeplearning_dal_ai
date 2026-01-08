import inspect
import os

from IPython.display import display, Markdown
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm




def display_code(function):
    """
    Displays the source code of a given function as a formatted markdown block.

    This is useful in environments like Jupyter Notebooks for inspecting the
    implementation of a function directly in the output.

    Args:
        function (callable): The function whose source code is to be displayed.
    """
    # Retrieve the source code of the function as a string.
    source_code = inspect.getsource(function)
    
    # Format the source code string as a python markdown block.
    markdown_code = f"```python\n{source_code}\n```"
    
    # Display the formatted code in the output.
    display(Markdown(markdown_code))
    
    
    
def training_loop_signature(model, train_loader, val_loader, loss_fcn, optimizer, threshold,
                            device, save_path=None, n_epochs=5, scheduler=None):
    """
    Executes the main training and evaluation loop in a single, self-contained function.

    This function orchestrates the training process over multiple epochs,
    handles validation, tracks the best model based on validation accuracy,
    and optionally saves the best performing model checkpoint.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained and evaluated.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        loss_fcn (callable): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        threshold (float): The distance threshold for classifying pairs during validation.
        device (torch.device): The device to run training on.
        save_path (str, optional): File path to save the best model. If None, the model
                                   is not saved. Defaults to None.
        n_epochs (int, optional): The total number of epochs to train. Defaults to 5.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.

    Returns:
        torch.nn.Module: The trained model. If save_path is provided, it returns the
                         model with the best validation weights loaded. Otherwise, it
                         returns the model from the final epoch.
    """
    # Ensure the directory for saving the model exists, if a path is provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
    # Move the model to the specified computing device
    model.to(device)
    best_val_acc = 0.0
    
    print("--- Starting Training & Validation ---\n")
    
    for epoch in range(1, n_epochs + 1):
        
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        train_samples_processed = 0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs} [Training]", leave=False)

        for data_batch in train_progress_bar:
            anchor, positive, negative = data_batch
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            anchor_out, positive_out, negative_out = model(anchor, positive, negative)
            loss = loss_fcn(anchor_out, positive_out, negative_out)
            
            loss.backward()
            optimizer.step()
            
            # Weight loss by batch size for correct averaging
            batch_size = anchor.size(0)
            running_train_loss += loss.item() * batch_size
            train_samples_processed += batch_size
            
            # Update the progress bar with the running average loss
            display_loss = running_train_loss / train_samples_processed
            train_progress_bar.set_postfix(loss=f'{display_loss:.4f}')
        
        train_loss = running_train_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        correct_predictions = 0
        total_pairs = 0
        running_val_loss = 0.0
        val_samples_processed = 0
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{n_epochs} [Validation]", leave=False)
        
        with torch.no_grad():
            for data_batch in val_progress_bar:
                anchor, positive, negative = data_batch
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                
                anchor_out, pos_out, neg_out = model(anchor, positive, negative)
                val_loss_item = loss_fcn(anchor_out, pos_out, neg_out)

                # Weight loss by batch size for correct averaging
                batch_size = anchor.size(0)
                running_val_loss += val_loss_item.item() * batch_size
                val_samples_processed += batch_size

                # Accuracy calculation
                dist_pos = F.pairwise_distance(anchor_out, pos_out)
                correct_predictions += torch.sum(dist_pos < threshold).item()
                dist_neg = F.pairwise_distance(anchor_out, neg_out)
                correct_predictions += torch.sum(dist_neg >= threshold).item()
                total_pairs += len(dist_pos) + len(dist_neg)

                # Update running metrics on the progress bar
                current_acc = correct_predictions / total_pairs if total_pairs > 0 else 0
                display_loss = running_val_loss / val_samples_processed
                val_progress_bar.set_postfix(acc=f'{current_acc:.2%}', loss=f'{display_loss:.4f}')

        val_accuracy = correct_predictions / total_pairs if total_pairs > 0 else 0
        val_loss = running_val_loss / len(val_loader.dataset)

        # Print a summary for the epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2%} | LR: {current_lr:.6f}\n")

        # Update the learning rate scheduler, if one is provided
        if scheduler:
            scheduler.step()
        
        # Save the model if it has the best validation accuracy so far
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"  -> New best model saved to '{save_path}' with Val Acc: {best_val_acc:.2%}\n")

    print("\n--- Training & Validation Complete ---")
    if save_path:
        print(f"Best model saved to '{save_path}' with accuracy {best_val_acc:.2%}")
    
    # Load the best performing model weights before returning, if a path was provided
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        
    return model



def training_loop_change(model, train_loader, val_loader, loss_fcn, optimizer, 
                         device, n_epochs=10, scheduler=None, save_path=None):
    """
    Executes a full training and validation loop for the change detection task.

    This function orchestrates the training process over multiple epochs,
    handles validation, tracks the best model based on validation loss,
    and saves the best performing model checkpoint to the specified path.

    Args:
        model (torch.nn.Module): The Siamese network model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        loss_fcn (callable): The loss function (e.g., WeightedContrastiveLoss).
        optimizer (torch.optim.Optimizer): The optimizer for updating weights.
        device (torch.device): The compute device ('cpu', 'cuda', or 'mps').
        n_epochs (int, optional): The total number of epochs for training. Defaults to 10.
        scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler.
        save_path (str, optional): File path to save the best performing model.
    
    Returns:
        torch.nn.Module: The trained model with the best validation weights loaded.
    """
    # Ensure the directory for saving the model exists
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
    # Move the model to the specified computing device
    model.to(device)
    best_val_loss = float('inf')

    print("--- Starting Training & Validation ---\n")
    
    for epoch in range(1, n_epochs + 1):
        
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        train_samples_processed = 0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs} [Training]", leave=False)

        for data_batch in train_progress_bar:
            before_imgs, after_imgs, labels = data_batch
            before_imgs, after_imgs, labels = before_imgs.to(device), after_imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            before_emb, after_emb = model(before_imgs, after_imgs, triplet_bool=False)
            loss = loss_fcn(before_emb, after_emb, labels)
            loss.backward()
            optimizer.step()
            
            # Weight loss by batch size for correct averaging
            batch_size = before_imgs.size(0)
            running_train_loss += loss.item() * batch_size
            train_samples_processed += batch_size
            
            # Update the progress bar with the running average loss
            display_loss = running_train_loss / train_samples_processed
            train_progress_bar.set_postfix(loss=f'{display_loss:.4f}')
        
        # Calculate the final, weighted average loss for the epoch
        train_loss = running_train_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        val_samples_processed = 0
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{n_epochs} [Validation]", leave=False)
        
        with torch.no_grad():
            for data_batch in val_progress_bar:
                before_imgs, after_imgs, labels = data_batch
                before_imgs, after_imgs, labels = before_imgs.to(device), after_imgs.to(device), labels.to(device)
                
                before_emb, after_emb = model(before_imgs, after_imgs, triplet_bool=False)
                val_loss_item = loss_fcn(before_emb, after_emb, labels)

                # Weight loss by batch size for correct averaging
                batch_size = before_imgs.size(0)
                running_val_loss += val_loss_item.item() * batch_size
                val_samples_processed += batch_size

                # Update running metrics on the progress bar
                display_loss = running_val_loss / val_samples_processed
                val_progress_bar.set_postfix(loss=f'{display_loss:.4f}')

        # Calculate final, weighted average loss for the epoch
        val_loss = running_val_loss / len(val_loader.dataset)

        # Print a summary for the epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}\n")

        # Update the learning rate scheduler
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save the model if it has the best validation loss so far
        if save_path and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved to '{save_path}' with Val Loss: {best_val_loss:.4f}\n")

    print("\n--- Training Complete ---")
    if save_path:
        print(f"Best model saved to '{save_path}' with validation loss {best_val_loss:.4f}")
    
    # Load the best performing model weights before returning
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        
    return model



def evaluation_loop(model, data_loader, loss_fcn, device):
    """
    Evaluates a trained model on a dataset to find and return the
    best-performing distance threshold.

    This function first runs the entire dataset through the model to collect
    all embedding distances and corresponding labels. It then iterates through a
    range of possible thresholds to find the one that yields the highest
    binary classification accuracy for 'Change' vs. 'No Change'.

    Args:
        model (nn.Module): The trained model to evaluate.
        data_loader (DataLoader): The DataLoader for the evaluation dataset.
        loss_fcn (callable): The loss function for calculating validation loss.
        device (torch.device): The device to run evaluation on.

    Returns:
        float: The optimal distance threshold that maximizes accuracy.
    """
    print("--- Starting Evaluation ---\n")
    
    # --- Setup ---
    model.eval()
    all_distances = []
    all_labels = []
    running_val_loss = 0.0

    # --- STEP 1: COLLECT ALL DISTANCES AND LABELS ---
    # Perform a full pass over the validation data to gather model outputs.
    progress_bar = tqdm(data_loader, desc="[Validation]", leave=False)
    with torch.no_grad():
        for data_batch in progress_bar:
            before_imgs, after_imgs, labels = data_batch
            before_imgs, after_imgs = before_imgs.to(device), after_imgs.to(device)
            labels = labels.to(device)

            before_emb, after_emb = model(before_imgs, after_imgs, triplet_bool=False)
            
            # Calculate and store distances and labels for later analysis.
            distances = F.pairwise_distance(before_emb, after_emb)
            all_distances.extend(distances.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Also calculate the overall validation loss.
            loss = loss_fcn(before_emb, after_emb, labels)
            running_val_loss += loss.item()

    avg_val_loss = running_val_loss / len(data_loader)
    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)

    # --- STEP 2: TEST CANDIDATE THRESHOLDS ---
    # Initialize variables to track the best performing threshold.
    best_accuracy = 0.0
    best_threshold = 0.0
    # Generate a range of potential thresholds to test across the observed distance range.
    thresholds = np.linspace(np.min(all_distances), np.max(all_distances), num=100)

    # --- STEP 3 & 4: CALCULATE ACCURACY AND IDENTIFY THE BEST THRESHOLD ---
    # Iterate through each candidate to find the one that maximizes accuracy.
    for t in thresholds:
        # Step 3: Calculate accuracy for the current threshold.
        # Predict 1 ('Change') if distance is greater than the threshold, otherwise 0 ('No Change').
        preds = (all_distances > t).astype(int)
        # Ground truth is 1 ('Change') if the label is not 2, otherwise 0 ('No Change').
        ground_truth = (all_labels != 2).astype(int)
        accuracy = np.mean(preds == ground_truth)

        # Step 4: Update the best threshold if the current one yields higher accuracy.
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = t
            
    # Print a summary of the evaluation results.
    print("\n--- Final Evaluation Results ---")
    print(f"  -> Best Binary Accuracy (Change/No Change): {best_accuracy:.2%}")
    print(f"  -> At Optimal Distance Threshold: {best_threshold:.4f}")
    print(f"  -> Validation Loss: {avg_val_loss:.4f}\n")
    
    # Return the best threshold found.
    return best_threshold