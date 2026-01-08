import glob
import os
import random
from collections import defaultdict

import cv2
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torchvision import models as tv_models
from tqdm.auto import tqdm


# --- Constants for Vegetation Detection ---
LOWER_GREEN_HSV = np.array([35, 40, 40])
UPPER_GREEN_HSV = np.array([85, 255, 255])



def _create_signature_map(base_data_dir, min_real=1, min_fake=1):
    """
    Scans the dataset directory to create a map of signatures for each ID.

    Args:
        base_data_dir (str): The root directory of the signature dataset.
        min_real (int): The minimum number of real images an ID must have.
        min_fake (int): The minimum number of fake images an ID must have.

    Returns:
        dict: A dictionary mapping valid user IDs to their image paths.
    """
    # Construct the path to the directory containing real signatures.
    real_signatures_dir = os.path.join(base_data_dir, 'Real')
    # Construct the path to the directory containing fake signatures.
    fake_signatures_dir = os.path.join(base_data_dir, 'Fake')
    # Initialize a dictionary that will map user IDs to their signature image paths.
    signature_map = defaultdict(lambda: {'real': [], 'fake': []})

    # Check if the directory for real signatures exists to prevent errors.
    if not os.path.isdir(real_signatures_dir):
        print(f"Error: Directory not found at: {real_signatures_dir}")
        return {}

    # Get and sort a list of all items in the real signatures directory.
    all_ids = sorted(os.listdir(real_signatures_dir))
    # Iterate through each potential user ID directory.
    for user_id in all_ids:
        # Process only directories that follow the expected user ID naming convention.
        if user_id.startswith('ID_'):
            # Find all real signature images for the current user.
            real_images = glob.glob(os.path.join(real_signatures_dir, user_id, '*.jpg'))
            
            # Define the path for the corresponding fake signatures directory.
            fake_user_dir = os.path.join(fake_signatures_dir, user_id)
            # Find all fake images if the user's fake directory exists, otherwise use an empty list.
            fake_images = glob.glob(os.path.join(fake_user_dir, '*.jpg')) if os.path.isdir(fake_user_dir) else []
            
            # Filter users based on the minimum required number of real and fake signatures.
            if len(real_images) >= min_real and len(fake_images) >= min_fake:
                # Store the list of real image paths for the valid user.
                signature_map[user_id]['real'] = real_images
                # Store the list of fake image paths for the valid user.
                signature_map[user_id]['fake'] = fake_images
    
    # Return the completed map of user IDs to their signature paths.
    return signature_map



def display_signature_dataset_summary(base_data_dir):
    """
    Displays a dynamic summary and bar charts of the dataset.
    """
    # Create a map of signatures, filtering by minimum image counts.
    signature_map = _create_signature_map(base_data_dir, min_real=2, min_fake=1)

    # Exit the function if no valid data was found.
    if not signature_map:
        print("No valid individuals found for triplet generation.")
        return

    # Convert the map to a list of dictionaries to prepare for DataFrame creation.
    data_list = [{'ID': user_id, 'Real': len(files['real']), 'Fake': len(files['fake'])} for user_id, files in signature_map.items()]
    # Create a pandas DataFrame from the list of data.
    df = pd.DataFrame(data_list)
    # Sort the DataFrame numerically by the ID number for consistent ordering.
    df_sorted = df.sort_values(by="ID", key=lambda x: x.str.split('_').str[1].astype(int)).reset_index(drop=True)

    # Calculate aggregate statistics for the summary.
    num_ids = len(df_sorted)
    total_real = df_sorted['Real'].sum()
    total_fake = df_sorted['Fake'].sum()

    # Display the high level summary statistics.
    print(f"Found {num_ids} valid IDs.")
    print(f"   - Total Real Images: {total_real}")
    print(f"   - Total Fake Images: {total_fake}\n")

    # Set a base font size for all plot elements for better readability.
    plt.rcParams['font.size'] = 14
    # Divide the sorted data into chunks for clearer visualization.
    chunks = {
        "IDs 1-17": df_sorted.iloc[0:17],
        "IDs 18-34": df_sorted.iloc[17:34],
        "IDs 35-51": df_sorted.iloc[34:51]
    }

    # Generate a separate plot for each chunk of data.
    for title, chunk in chunks.items():
        # Skip empty chunks to avoid creating empty plots.
        if chunk.empty:
            continue

        # Extract data series from the chunk for plotting.
        ids = chunk['ID']
        real_counts = chunk['Real']
        fake_counts = chunk['Fake']

        # Define the label locations for the x axis.
        x = np.arange(len(ids))
        # Define the width of the bars in the bar chart.
        width = 0.35

        # Create a new figure and axes for the plot.
        fig, ax = plt.subplots(figsize=(10, 8))
        # Plot the bars for real and fake signature counts.
        rects1 = ax.bar(x - width/2, real_counts, width, label='Real Images', color='royalblue')
        rects2 = ax.bar(x + width/2, fake_counts, width, label='Fake Images', color='salmon')

        # Customize plot elements for clarity.
        ax.set_ylabel('Image Count')
        ax.set_title(f'Signature Counts for {title}', fontsize=18, weight='bold')
        ax.set_xticks(x)
        # Rotate x axis labels to prevent overlap.
        ax.set_xticklabels(ids, rotation=45, ha="right")
        ax.legend()
        
        # Add numeric labels on top of each bar for precise values.
        ax.bar_label(rects1, padding=3, fontsize=12)
        ax.bar_label(rects2, padding=3, fontsize=12)

        # Adjust layout to prevent plot elements from being cut off.
        fig.tight_layout()
        # Display the generated plot.
        plt.show()
        


def display_random_signature_pair(base_data_dir):
    """
    Displays a random pair of real and fake signatures.
    """
    # Create a map of signatures, filtering for IDs with at least one of each type.
    signature_map = _create_signature_map(base_data_dir, min_real=1, min_fake=1)

    # Exit the function if no valid data was found.
    if not signature_map:
        print("No valid individuals with both real and fake signatures were found.")
        return
    
    # Randomly select one individual from the list of valid IDs.
    random_id = random.choice(list(signature_map.keys()))
    
    # Get the lists of available images for the selected individual.
    real_images = signature_map[random_id]['real']
    fake_images = signature_map[random_id]['fake']
    
    # Randomly select one real and one fake signature path from the lists.
    real_image_path = random.choice(real_images)
    fake_image_path = random.choice(fake_images)
    
    # Attempt to load the selected images from their respective paths.
    try:
        real_img = Image.open(real_image_path)
        fake_img = Image.open(fake_image_path)
    # Handle cases where an image file cannot be found.
    except FileNotFoundError as e:
        print(f"Error: Could not find image file. {e}")
        return

    # Create a figure with two subplots side by side for comparison.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    # Set a main title for the entire figure.
    fig.suptitle(f'Signature Comparison for {random_id}', fontsize=16)
    
    # Display the real signature on the left subplot.
    ax1.imshow(real_img, cmap='gray')
    ax1.set_title('Real Signature')
    # Turn off the axis for a cleaner look.
    ax1.axis('off')
    
    # Display the fake signature on the right subplot.
    ax2.imshow(fake_img, cmap='gray')
    ax2.set_title('Fake Signature')
    # Turn off the axis for a cleaner look.
    ax2.axis('off')
    
    # Render the final plot to the screen.
    plt.show()
    

     
def create_signature_datasets_splits(full_dataset, train_split, train_transform, val_transform):
    """
    Splits a pre initialized dataset into training and validation sets,
    applying different transformations to each.

    Args:
        full_dataset: The complete dataset object, initialized with transform=None.
        train_split: The proportion of the dataset to use for training (e.g., 0.8).
        train_transform: Transformations for the training set.
        val_transform: Transformations for the validation set.

    Returns:
        tuple: A tuple containing the (train_dataset, val_dataset).
    """
    
    # Define an internal wrapper class to apply a specific transform to a dataset subset.
    class TransformedSignatureSubset(Dataset):
        # Initialize the subset with its data and a specific transformation pipeline.
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        
        # Retrieve an item by index and apply the transformation.
        def __getitem__(self, index):
            # Get the untransformed data triplet from the base subset.
            anchor, positive, negative = self.subset[index]
            
            # Apply the stored transformation pipeline to each image if it exists.
            if self.transform:
                anchor = self.transform(anchor)
                positive = self.transform(positive)
                negative = self.transform(negative)
            
            # Return the transformed triplet.
            return anchor, positive, negative
            
        # Return the total number of items in the subset.
        def __len__(self):
            return len(self.subset)

    # Ensure the provided dataset was created without a default transform.
    if full_dataset.transform is not None:
        raise ValueError("The 'full_dataset' must be initialized with transform=None for this function to work correctly.")

    # Calculate the number of samples for the training and validation sets.
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    # Perform a random split of the original dataset into two subsets.
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Wrap each new subset with its corresponding transformation pipeline.
    train_dataset = TransformedSignatureSubset(train_subset, train_transform)
    val_dataset = TransformedSignatureSubset(val_subset, val_transform)

    # Return the final training and validation datasets.
    return train_dataset, val_dataset


      
def deprocess_signature_image(tensor):
    """
    Reverses the normalization on an image tensor for display.
    """
    # Define the mean and standard deviation used for the original normalization.
    mean = np.array([0.861, 0.861, 0.861])
    std = np.array([0.274, 0.274, 0.274])
    
    # Convert the tensor to a NumPy array in the proper format for image processing.
    tensor = tensor.clone().detach().cpu().numpy().transpose(1, 2, 0)
    
    # Reverse the normalization process (denormalize).
    tensor = std * tensor + mean
    
    # Clip the values to the valid range [0, 1] to prevent display errors.
    tensor = np.clip(tensor, 0, 1)
    
    # Return the processed image array ready for visualization.
    return tensor



def show_random_triplet(dataloader):
    """
    Displays a single random triplet from a DataLoader.
    """
    # Check if the dataloader is valid before proceeding.
    if not dataloader:
        print("DataLoader is not available. Cannot display triplet.")
        return

    # Retrieve a single batch of data from the iterator.
    anchor_batch, positive_batch, negative_batch = next(iter(dataloader))
    
    # Select the first triplet from the batch for display.
    anchor = anchor_batch[0]
    positive = positive_batch[0]
    negative = negative_batch[0]
    
    # Prepare the image tensors for visualization by reversing normalization.
    anchor_img = deprocess_signature_image(anchor)
    positive_img = deprocess_signature_image(positive)
    negative_img = deprocess_signature_image(negative)
    
    # Create a figure and a set of subplots for displaying the images.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # Set a main title for the entire figure.
    fig.suptitle('Example of a Training Triplet', fontsize=16)
    
    # Display the anchor image in the first subplot.
    axes[0].imshow(anchor_img)
    axes[0].set_title('Anchor (Real)')
    axes[0].axis('off')
    
    # Display the positive image in the second subplot.
    axes[1].imshow(positive_img)
    axes[1].set_title('Positive (Real)')
    axes[1].axis('off')
    
    # Display the negative image in the third subplot.
    axes[2].imshow(negative_img)
    axes[2].set_title('Negative (Fake of Same ID)')
    axes[2].axis('off')
        
    # Adjust subplot parameters for a tight layout.
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Render the plot to the screen.
    plt.show()
    
    
    
def show_signature_val_predictions(model, val_loader, threshold, device):
    """
    Visualizes model performance on a sample from the validation set.

    This function fetches a single batch, randomly selects one sample from it,
    and creates two plots: one for a genuine pair (anchor vs. positive) and
    one for a forgery pair (anchor vs. negative). Each plot displays the
    images, the model's prediction, and a graphical bar chart comparing the
    calculated distance to the decision threshold.

    Args:
        model (torch.nn.Module): The trained Siamese network model.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        threshold (float): The distance threshold for classifying pairs.
        device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').
    """
    print("--- Displaying Validation Predictions with Distance Visualization ---\n")
    # Set the model to evaluation mode to disable layers like dropout
    model.eval()
    
    # Safely get a single batch of data from the loader
    try:
        anchor, positive, negative = next(iter(val_loader))
    except StopIteration:
        print("DataLoader is empty.")
        return
        
    # Move image tensors to the specified computation device
    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
    
    # Safeguard against empty or small batches
    batch_size = anchor.size(0)
    if batch_size < 1:
        print("Batch is empty, cannot show examples.")
        return
    
    # Randomly select one index from the batch to create visualization pairs from
    index = random.choice(range(batch_size))
    
    # Prepare data for one genuine and one forgery pair for display
    # Each tuple contains: (image1, image2, pair_type, title1, title2, true_label)
    pairs_to_show = [
        (anchor[index], positive[index], "Genuine Pair", "Real 1", "Real 2", 1),
        (anchor[index], negative[index], "Forgery Pair", "Real", "Fake", 0)
    ]
    
    # Iterate through the prepared pairs to generate and display predictions
    for img1_tensor, img2_tensor, pair_type, title1, title2, true_label in pairs_to_show:
        # Disable gradient calculations for inference
        with torch.no_grad():
            # Generate embeddings for both images in the pair
            emb1 = model.get_embedding(img1_tensor.unsqueeze(0))
            emb2 = model.get_embedding(img2_tensor.unsqueeze(0))
            # Calculate the Euclidean distance between the two embeddings
            distance = F.pairwise_distance(emb1, emb2).item()
            # Make a prediction based on whether the distance is below the threshold
            prediction = 1 if distance < threshold else 0

        # --- Plotting Setup ---
        # Create a figure and a custom grid layout for the plots
        fig = plt.figure(figsize=(8, 5.5))
        gs = fig.add_gridspec(2, 2, height_ratios=[4, 1]) # Top row for images, bottom for bar plot

        ax1 = fig.add_subplot(gs[0, 0])      # Top-left for image 1
        ax2 = fig.add_subplot(gs[0, 1])      # Top-right for image 2
        ax_dist = fig.add_subplot(gs[1, :])  # Bottom row (spanned) for the distance plot

        # Prepare dynamic strings and colors based on the prediction's correctness
        prediction_str = 'Genuine' if prediction == 1 else 'Forgery'
        result_str = 'CORRECT' if prediction == true_label else 'INCORRECT'
        title_color = 'green' if prediction == true_label else 'red'
        bar_color = 'green' if prediction == true_label else 'red'
        operator_str = '<' if distance < threshold else '>='
        
        # Construct the multi-line title for the entire figure
        title = (f"Type: {pair_type}\n"
                 f"Prediction: {prediction_str} -> {result_str}\n"
                 f"Dist: {distance:.2f} {operator_str} Thresh: {threshold:.2f}")
        fig.suptitle(title, color=title_color, fontsize=12, y=0.98)
        
        # --- Display Images (Top Row) ---
        ax1.imshow(deprocess_signature_image(img1_tensor))
        ax1.set_title(title1)
        ax1.axis('off')
        
        ax2.imshow(deprocess_signature_image(img2_tensor))
        ax2.set_title(title2)
        ax2.axis('off')

        # --- Display Distance Plot (Bottom Row) ---
        # Plot the calculated distance as a horizontal bar
        ax_dist.barh([0], [distance], color=bar_color, height=0.5, zorder=1)
        # Draw the threshold as a vertical line on top of the bar
        ax_dist.axvline(x=threshold, color='black', linestyle='--', zorder=2)
        # Add a text label showing the precise distance value
        ax_dist.text(distance + 0.02, 0, f'{distance:.2f}', va='center', fontweight='bold')
        
        # Format the distance plot for clarity
        ax_dist.set_yticks([]) # Hide y-axis ticks
        ax_dist.set_xlim(0, max(distance, threshold) * 1.5) # Set dynamic x-axis limit
        ax_dist.set_xlabel("Euclidean Distance in Embedding Space")
        
        # Adjust layout and display the final plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.92]) 
        plt.show()
        
        
        
def verify_signature(model, genuine_path, test_path, threshold, transform, device):
    """
    Performs one shot verification for a pair of signatures.
    
    Args:
        model: The trained Siamese network.
        genuine_path (str): Path to a known genuine signature (the anchor).
        test_path (str): Path to the signature being tested.
        threshold (float): The optimal decision threshold from evaluation.
        transform: The image transformations.
        device: The device to run on (CPU or GPU).
    """
    # Set the model to evaluation mode to disable layers like dropout.
    model.eval() 
    
    # Load and process both the genuine and test images.
    try:
        # Apply transformations and move the genuine image tensor to the correct device.
        img_genuine = transform(Image.open(genuine_path).convert("RGB")).unsqueeze(0).to(device)
        # Apply transformations and move the test image tensor to the correct device.
        img_test = transform(Image.open(test_path).convert("RGB")).unsqueeze(0).to(device)
    except FileNotFoundError as e:
        print(f"Error loading image: {e}")
        return

    # Disable gradient calculations for inference to save memory and computations.
    with torch.no_grad():
        # Generate embeddings for both images using the model.
        emb_genuine = model.get_embedding(img_genuine)
        emb_test = model.get_embedding(img_test)
        
        # Calculate the euclidean distance between the two embeddings.
        distance = F.pairwise_distance(emb_genuine, emb_test).item()
        
        # Make a prediction based on whether the distance is below the threshold.
        is_genuine = distance < threshold
        
    # Display the numerical results of the verification.
    print(f"--- Verification Result ---")
    print(f"Distance: {distance:.4f}")
    print(f"Decision Threshold: {threshold:.4f}")
    # Print the final prediction outcome.
    if is_genuine:
        print("Prediction: ✅ Genuine Signature\n")
    else:
        print("Prediction: ❌ Forgery Detected\n")
        
    # Create a figure with two subplots for visual comparison.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    # Set the main title of the figure to show the calculated distance.
    fig.suptitle(f'Distance: {distance:.4f}', fontsize=16)
    
    # Display the known genuine signature in the left subplot.
    ax1.imshow(Image.open(genuine_path))
    ax1.set_title("Known Genuine Signature")
    ax1.axis('off')
    
    # Display the signature to be verified in the right subplot.
    ax2.imshow(Image.open(test_path))
    ax2.set_title("Signature to Verify")
    ax2.axis('off')
    
    # Render the final plot.
    plt.show()
    
    

def _create_change_map(base_data_dir):
    """
    Scans the change detection dataset directory and creates a map of
    'Before' and 'After' image pairs for each change category.

    Args:
        base_data_dir (str): The root directory of the change detection dataset.

    Returns:
        dict: A dictionary mapping change categories to a list of
              (before_path, after_path) tuples.
    """
    # Initialize a dictionary to store the paths for each category.
    change_map = {'Positive': [], 'Negative': [], 'No_Change': []}
    # Get the list of category names to iterate over.
    categories = list(change_map.keys())
    
    # Loop through each category directory (Positive, Negative, No_Change).
    for category in categories:
        # Construct the full paths for the category's subdirectories.
        category_path = os.path.join(base_data_dir, category)
        before_dir = os.path.join(category_path, 'Before')
        after_dir = os.path.join(category_path, 'After')

        # Skip the current category if its 'Before' directory does not exist.
        if not os.path.isdir(before_dir):
            continue

        # Iterate through all files found in the 'Before' directory.
        for filename in os.listdir(before_dir):
            # Construct the full file path for the 'before' and corresponding 'after' image.
            before_path = os.path.join(before_dir, filename)
            after_path = os.path.join(after_dir, filename)
            
            # Check if the corresponding 'after' image exists to form a valid pair.
            if os.path.exists(after_path):
                # If the pair is valid, add the tuple of paths to the map.
                change_map[category].append((before_path, after_path))
    
    # Return the completed map of image pairs.
    return change_map


    
    
def display_change_dataset_stats(base_data_dir):
    """
    Displays a detailed statistical summary of the change detection dataset.

    This function processes the dataset found in the specified directory,
    calculates the number of image pairs for each change category, and
    presents the statistics in a styled table format.

    Args:
        base_data_dir (str): The root directory of the change detection dataset.
    """
    # Generate a map of change categories to corresponding image pairs.
    change_map = _create_change_map(base_data_dir)

    # Verify that the change map is not empty before proceeding.
    if not any(change_map.values()):
        print("The change map is empty. Cannot display stats.")
        return

    # Restructure the map data into a list of dictionaries for DataFrame creation.
    data_list = []
    for category, pairs in change_map.items():
        data_list.append({
            'Change Category': category,
            'Number of Image Pairs': len(pairs)
        })

    # Create a DataFrame from the list, sort it by category, and reset the index.
    df = pd.DataFrame(data_list).sort_values(by='Change Category').reset_index(drop=True)

    # Calculate the total number of image pairs across all categories.
    total_pairs = df['Number of Image Pairs'].sum()
    
    # Create a new DataFrame for the summary 'Total' row.
    total_row = pd.DataFrame([{
        'Change Category': '<b>Total</b>',
        'Number of Image Pairs': total_pairs
    }])
    
    # Append the total row to the main DataFrame for display.
    df_display = pd.concat([df, total_row], ignore_index=True)

    # Initialize a Styler object to customize the DataFrame's appearance, starting by hiding the index.
    styler = df_display.style.hide(axis="index")
    
    # Apply custom CSS styles to the table elements for improved readability.
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
    # Set additional properties for table cells to ensure content wraps properly.
    styler.set_properties(**{"white-space": "normal"})
    
    # Render the styled DataFrame in the output.
    display(styler)
    
    
 
    
def display_random_change_pairs(base_data_dir):
    """
    Displays a random 'Before' and 'After' image pair from each change
    category, with the category name centered above each pair.

    This function generates a visual comparison for 'Positive', 'Negative',
    and 'No_Change' categories by selecting a random image pair from each
    and arranging them in a clear, titled grid layout.

    Args:
        base_data_dir (str): The root directory of the change detection dataset.
    """
    # Generate a map of change categories to their corresponding image pairs.
    change_map = _create_change_map(base_data_dir)

    # Verify that the change map is not empty before attempting to display images.
    if not any(change_map.values()):
        print("The change map is empty. Cannot display image pairs.")
        return

    # Initialize the main figure for plotting.
    fig = plt.figure(figsize=(8, 15))

    # Create an outer grid to structure the plots for each category vertically.
    outer_grid = GridSpec(3, 1, figure=fig, hspace=0.2)
    
    # Define the specific order of categories to be displayed.
    categories = ['Positive', 'Negative', 'No_Change']

    # Loop through each category to create and display its corresponding plot.
    for i, category in enumerate(categories):
        # Format the category name for a cleaner display title.
        display_category = category.replace('_', ' ')
        # Retrieve the list of image pairs for the current category.
        image_pairs = change_map.get(category, [])
        
        # Create a nested grid within the outer grid for the category title and the image pair.
        # The height ratio gives more space to the images compared to the title.
        inner_grid = outer_grid[i].subgridspec(2, 2, height_ratios=[0.05, 1], wspace=0.05, hspace=0.1)

        # Add a subplot for the category title, spanning both columns of the inner grid.
        ax_title = fig.add_subplot(inner_grid[0, :])
        # Add and style the category title text.
        ax_title.text(0.5, 0.5, display_category, ha='center', va='center', fontsize=18, weight='bold')
        # Hide the axes for the title subplot.
        ax_title.axis('off')

        # Create subplots for the 'Before' and 'After' images.
        ax_before = fig.add_subplot(inner_grid[1, 0])
        ax_after = fig.add_subplot(inner_grid[1, 1])

        # Set the titles for the individual image subplots.
        ax_before.set_title('Before', fontsize=16)
        ax_after.set_title('After', fontsize=16)
        
        # Check if any image pairs exist for the current category.
        if not image_pairs:
            # Display a message if no images are found.
            ax_before.text(0.5, 0.5, 'No Images Found', ha='center', va='center', fontsize=12)
            ax_after.text(0.5, 0.5, 'No Images Found', ha='center', va='center', fontsize=12)
        else:
            # Randomly select one 'Before' and 'After' image pair.
            before_path, after_path = random.choice(image_pairs)
            # Attempt to open and display images, handling potential file errors.
            try:
                ax_before.imshow(Image.open(before_path))
                ax_after.imshow(Image.open(after_path))
            except FileNotFoundError:
                # Display a message if an image file cannot be found.
                ax_before.text(0.5, 0.5, 'Image Not Found', ha='center', va='center', fontsize=12)
                ax_after.text(0.5, 0.5, 'Image Not Found', ha='center', va='center', fontsize=12)

        # Turn off the axes for the image plots for a cleaner look.
        ax_before.axis('off')
        ax_after.axis('off')

    # Render and display the final composite figure.
    plt.show()
    
    


def create_change_datasets_splits(full_dataset, train_split, train_transform, val_transform):
    """
    Splits a pre-initialized change detection dataset into training and
    validation sets, applying different transformations to each.

    This function is designed to take a single, untransformed dataset and
    produce reproducible training and validation splits, each with its own
    set of data augmentation and preprocessing transforms.

    Args:
        full_dataset (Dataset): The complete dataset object, initialized with transform=None.
        train_split (float): The proportion of the dataset to allocate for training (e.g., 0.8).
        train_transform (callable): The transformations to apply to the training set.
        val_transform (callable): The transformations to apply to the validation set.

    Returns:
        tuple: A tuple containing the created (train_dataset, val_dataset).
        
    Raises:
        ValueError: If the `full_dataset` is initialized with any transforms.
    """
    
    # Define an internal wrapper class to apply a specific transform to a dataset subset.
    class TransformedChangeSubset(Dataset):
        def __init__(self, subset, transform):
            # Store the subset of data (e.g., from random_split).
            self.subset = subset
            # Store the transformations to be applied to this specific subset.
            self.transform = transform
        
        def __getitem__(self, index):
            # Retrieve the untransformed data item from the original subset.
            before_img, after_img, label = self.subset[index]
            
            # Apply the specified transformations to both images if a transform is provided.
            if self.transform:
                before_img = self.transform(before_img)
                after_img = self.transform(after_img)
            
            return before_img, after_img, label
            
        def __len__(self):
            # Return the total number of items in the subset.
            return len(self.subset)

    # Validate that the input dataset has not already been assigned a transform.
    if full_dataset.transform is not None:
        raise ValueError("The 'full_dataset' must be initialized with transform=None for this function to work correctly.")

    # Calculate the exact number of samples for the training and validation splits.
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    # Create a generator with a fixed seed for reproducible splits.
    generator = torch.Generator().manual_seed(42)
    # Perform a random split of the dataset into training and validation subsets.
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Wrap each subset with the custom class to apply the correct set of transforms.
    train_dataset = TransformedChangeSubset(train_subset, train_transform)
    val_dataset = TransformedChangeSubset(val_subset, val_transform)

    return train_dataset, val_dataset


 
def deprocess_change_image(tensor):
    """
    Reverses ImageNet normalization on an image tensor for display.

    This function takes a normalized PyTorch tensor, converts it back to a
    displayable image format by reversing the normalization, and scales it
    to the standard 0-255 pixel value range.

    Args:
        tensor (torch.Tensor): A PyTorch tensor representing an image,
                               typically with shape (C, H, W).

    Returns:
        np.ndarray: A NumPy array representing the deprocessed image in
                    (H, W, C) format with pixel values in the range [0, 255].
    """
    # Define the mean and standard deviation used for ImageNet normalization.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Convert the tensor to a NumPy array and change channel order from (C, H, W) to (H, W, C).
    tensor = tensor.clone().detach().cpu().numpy().transpose(1, 2, 0)

    # Reverse the normalization process (de-standardize).
    tensor = std * tensor + mean

    # Clip the values to ensure they are within the valid [0, 1] range.
    tensor = np.clip(tensor, 0, 1)

    # Scale the pixel values from [0, 1] to [0, 255] and convert to an 8-bit integer.
    return (tensor * 255).astype(np.uint8)




def show_random_pair(dataloader):
    """
    Displays a single random 'Before' and 'After' pair from a DataLoader.

    This function fetches one batch from the provided DataLoader, selects the
    first item, de-processes the image tensors for visualization, and plots
    them side-by-side with appropriate titles.

    Args:
        dataloader: A DataLoader object that yields batches of
                    (before_image, after_image, label) tuples.
    """
    # Ensure the DataLoader is valid before proceeding.
    if not dataloader:
        print("DataLoader is not available. Cannot display a pair.")
        return

    # Retrieve a single batch of data from the dataloader.
    before_batch, after_batch, label_batch = next(iter(dataloader))
    
    # Select the first image pair and its corresponding label from the batch.
    before_tensor = before_batch[0]
    after_tensor = after_batch[0]
    label = label_batch[0].item()
    
    # Convert the image tensors from their normalized format to a displayable format.
    before_img = deprocess_change_image(before_tensor)
    after_img = deprocess_change_image(after_tensor)
    
    # Map the numerical label to its human-readable string representation for the title.
    class_map = {0: 'Positive', 1: 'Negative', 2: 'No Change'}
    class_name = class_map.get(label, 'Unknown')
    
    # Create a plot to display the 'Before' and 'After' images side-by-side.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f'Example of a "{class_name}" Training Pair', fontsize=16)
    
    # Display the 'Before' image.
    ax1.imshow(before_img)
    ax1.set_title('Before')
    ax1.axis('off')
    
    # Display the 'After' image.
    ax2.imshow(after_img)
    ax2.set_title('After')
    ax2.axis('off')
          
    # Adjust layout to prevent titles from overlapping and display the plot.
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
 
    
def compute_change_class_weights(train_dataset, full_untransformed_dataset):
    """
    Computes class weights for the training set to handle class imbalance.

    Args:
        train_dataset (Dataset): The training subset, expected to be a wrapper
                                 around a `torch.utils.data.Subset` instance.
        full_untransformed_dataset (Dataset): The original, complete dataset from
                                              which the subset was created.

    Returns:
        torch.Tensor: A tensor of weights for each class, suitable for use in a
                      loss function like `nn.CrossEntropyLoss`.
    """
    # Get the list of indices that belong to the training subset.
    train_indices = train_dataset.subset.indices

    # Extract the ground truth labels for the training samples from the original full dataset.
    train_labels = [full_untransformed_dataset.image_pairs[i][2] for i in train_indices]

    # Use scikit-learn's utility to compute weights that balance the classes.
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )

    # Convert the NumPy array of weights into a PyTorch tensor.
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    return weights_tensor
    
    
    
def get_efficientnet_embedding_backbone(embedding_dim=128, pretrained=True,
    weights_path="./pretrained_efficientnet_weights/efficientnet_b0_rwightman-7f5810bc.pth"):
    """
    Builds an embedding network from an EfficientNet-B0 model, loading weights offline.

    This function configures an EfficientNet-B0 model to act as a feature extractor
    by replacing its final classification layer with a linear layer that outputs
    an embedding of a specified dimension. It supports loading weights from a local file.

    Args:
        embedding_dim (int): The desired dimension of the output embedding vector.
        pretrained (bool): If True, loads weights from the local `weights_path`.
        weights_path (str): The local file path to the .pth weights file.

    Returns:
        torch.nn.Module: The modified EfficientNet-B0 model.
        
    Raises:
        FileNotFoundError: If `pretrained` is True and the weights file is not found.
    """
    # Instantiate the EfficientNet-B0 model architecture with random weights.
    model = tv_models.efficientnet_b0(weights=None)

    # If specified, load the pretrained weights from a local file.
    if pretrained:
        # Raise an error if the weights file is missing to prevent silent failures.
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at the specified path: {weights_path}")
        
        # Load the state dictionary from the path, mapping to CPU to avoid device errors.
        state_dict = torch.load(weights_path, map_location='cpu')
        # Load the state dictionary into the model.
        model.load_state_dict(state_dict)
    
    # Modify the final classification layer to produce an embedding.
    # Get the number of input features from the original classifier layer.
    num_ftrs = model.classifier[1].in_features
    # Replace the layer with a new linear layer of the desired embedding dimension.
    model.classifier[1] = nn.Linear(num_ftrs, embedding_dim)
    
    return model



def calculate_vegetation_percentage(image_array):
    """
    Calculates the percentage of an image that contains green vegetation.

    This function identifies vegetation by converting the input image to the
    HSV color space and creating a mask for pixels that fall within a
    pre-defined range for the color green.

    Args:
        image_array (np.ndarray): A NumPy array representing an RGB image,
                                  with shape (height, width, 3).

    Returns:
        float: The percentage of the image classified as vegetation,
               represented as a value between 0.0 and 1.0.
    """
    # Convert the input RGB image to the HSV (Hue, Saturation, Value) color space.
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

    # Create a binary mask where green pixels are white (255) and others are black (0).
    mask = cv2.inRange(hsv_image, LOWER_GREEN_HSV, UPPER_GREEN_HSV)

    # Count the number of pixels identified as vegetation (white pixels in the mask).
    veg_count = np.sum(mask == 255)

    # Calculate the total number of pixels in the image.
    total_pixels = image_array.shape[0] * image_array.shape[1]

    # Return the ratio of vegetation pixels to total pixels, handling division by zero.
    return veg_count / total_pixels if total_pixels > 0 else 0




def classify_greenery_change(input_before, input_after, threshold_percent):
    """
    Analyzes two images and classifies the change in greenery.

    This function can process inputs as either file paths or PyTorch tensors.
    It calculates the percentage of green vegetation in each image and
    determines if the change between them is positive, negative, or negligible
    based on a given threshold.

    Args:
        input_before (str or torch.Tensor): The 'before' image.
        input_after (str or torch.Tensor): The 'after' image.
        threshold_percent (float): The percentage change required to be
                                   considered significant (e.g., 5.0 for 5%).

    Returns:
        str: A classification label: 'Positive', 'Negative', or 'No_Change'.
             Returns an error string if a file path is not found.

    Raises:
        TypeError: If inputs are not of type str or torch.Tensor.
    """
    img_before_array = None
    img_after_array = None

    # Handle input based on whether it is a file path or a tensor.
    if isinstance(input_before, str):
        # If input is a string, treat it as a file path and load the image.
        try:
            img_before_array = np.array(Image.open(input_before).convert("RGB"))
            img_after_array = np.array(Image.open(input_after).convert("RGB"))
        except FileNotFoundError:
            return 'Error: File not found'
            
    elif isinstance(input_before, torch.Tensor):
        # If input is a tensor, de-process it into a NumPy array.
        img_before_array = deprocess_change_image(input_before)
        img_after_array = deprocess_change_image(input_after)
        
    else:
        # Raise an error for unsupported input types.
        raise TypeError("Inputs must be either file paths (str) or PyTorch tensors.")

    # Calculate the percentage of vegetation in each image.
    percent_before = calculate_vegetation_percentage(img_before_array)
    percent_after = calculate_vegetation_percentage(img_after_array)

    # Determine the difference in vegetation and convert the threshold to a decimal.
    veg_change = percent_after - percent_before
    threshold_decimal = threshold_percent / 100.0

    # Classify the change based on the calculated difference and threshold.
    if veg_change >= threshold_decimal:
        return 'Positive'
    elif veg_change <= -threshold_decimal:
        # For a negative change, ensure the initial vegetation was significant.
        if percent_before >= threshold_decimal:
            return 'Negative'
        else:
            return 'No_Change'
    else:
        # If the change is within the threshold, it is considered no change.
        return 'No_Change'
    
    
    
def show_change_val_predictions(model, val_loader, model_threshold, device):
    """
    Displays one random example from each class (Positive, Negative, No_Change)
    from the validation set, showing the model's prediction for each.

    This function iterates through the validation set, selects one random sample
    for each class, runs them through the model to get a prediction, and then
    visualizes the 'before' and 'after' images along with the true and
    predicted labels.

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        val_loader (DataLoader): The DataLoader for the validation set.
        model_threshold (float): The distance threshold for classifying change.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to run inference on.
    """
    print(f"--- Displaying One Random Prediction for Each Class (Model Threshold: {model_threshold:.4f}) ---\n")
    model.eval()
    
    # First, collect all validation samples and sort them by class.
    positive_samples, negative_samples, no_change_samples = [], [], []
    for before_batch, after_batch, label_batch in val_loader:
        for i in range(len(label_batch)):
            label = label_batch[i].item()
            sample = {
                "before_tensor": before_batch[i],
                "after_tensor": after_batch[i],
                "true_label_int": label,
            }
            if label == 0: positive_samples.append(sample)
            elif label == 1: negative_samples.append(sample)
            else: no_change_samples.append(sample)

    # Ensure samples from all classes were found before proceeding.
    if not all([positive_samples, negative_samples, no_change_samples]):
        print("Validation set is missing samples from one or more classes.")
        return

    # Randomly select one sample from each class list to display.
    samples_to_show = [
        random.choice(positive_samples),
        random.choice(negative_samples),
        random.choice(no_change_samples)
    ]
    
    # Map integer labels to their string representations for plotting.
    class_map = {0: 'Positive', 1: 'Negative', 2: 'No Change'}

    # Process and display each of the three selected samples.
    for sample in samples_to_show:
        before_tensor = sample["before_tensor"].to(device)
        after_tensor = sample["after_tensor"].to(device)
        true_label_int = sample["true_label_int"]

        # Perform inference without calculating gradients for efficiency.
        with torch.no_grad():
            emb1, emb2 = model(before_tensor.unsqueeze(0), after_tensor.unsqueeze(0), triplet_bool=False)
            distance = F.pairwise_distance(emb1, emb2).item()

            # Make a prediction using a two-step process:
            # 1. Use the distance to predict 'Change' vs. 'No Change'.
            # 2. If 'Change' is predicted, use a heuristic to guess the change type.
            if distance <= model_threshold:
                final_prediction_int = 2
            else:
                change_type_str = classify_greenery_change(
                    input_before=before_tensor,
                    input_after=after_tensor,
                    threshold_percent=5.0
                )
                if change_type_str == 'Positive':
                    final_prediction_int = 0
                else:
                    final_prediction_int = 1
        
        # Prepare display strings and colors based on the prediction outcome.
        true_label_str = class_map.get(true_label_int, "Unknown")
        final_prediction_str = class_map.get(final_prediction_int, "Unknown")
        result_str = 'CORRECT' if final_prediction_int == true_label_int else 'INCORRECT'
        title_color = 'green' if final_prediction_int == true_label_int else 'red'
        
        # Construct a detailed title with the results and distance comparison.
        comparison_op = "<=" if distance <= model_threshold else ">"
        title = (f"True Label: {true_label_str} | Prediction: {final_prediction_str} -> {result_str}\n"
                 f"Distance: {distance:.2f} {comparison_op} {model_threshold:.2f} (Threshold)")

        # Create the plot for the 'Before' and 'After' images.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(title, color=title_color, fontsize=12, y=1.03)
        
        ax1.imshow(deprocess_change_image(before_tensor))
        ax1.set_title('Before')
        ax1.axis('off')
        
        ax2.imshow(deprocess_change_image(after_tensor))
        ax2.set_title('After')
        ax2.axis('off')
        
        plt.show()
        
        
        
        
def predict_greenery_change(model, before_path, after_path, model_threshold, transform, device):
    """
    Performs one-shot change detection for a pair of images.

    This function loads, preprocesses, and runs a pair of images through a
    trained Siamese network. It then classifies the change and displays the
    images along with the prediction result.

    Args:
        model (torch.nn.Module): The trained Siamese network.
        before_path (str): Path to the 'before' image.
        after_path (str): Path to the 'after' image.
        model_threshold (float): The optimal decision threshold from evaluation.
        transform (callable): The image transformations to apply.
        device (torch.device): The device to run inference on (e.g., 'cpu' or 'cuda').
        
    Returns:
        None. Prints the prediction and displays a plot.
    """
    # Set the model to evaluation mode to disable layers like dropout.
    model.eval()
    
    # Load and preprocess the images, handling potential file errors.
    try:
        img_before = transform(Image.open(before_path).convert("RGB")).unsqueeze(0).to(device)
        img_after = transform(Image.open(after_path).convert("RGB")).unsqueeze(0).to(device)
    except FileNotFoundError as e:
        print(f"Error loading image: {e}")
        return

    # Perform inference without calculating gradients for efficiency.
    with torch.no_grad():
        # Get the image embeddings from the model.
        emb_before, emb_after = model(img_before, img_after, triplet_bool=False)
        # Calculate the distance between the two embeddings.
        distance = F.pairwise_distance(emb_before, emb_after).item()
        
        # Use the distance and threshold to make a prediction.
        if distance <= model_threshold:
            # If the distance is below the threshold, classify as no significant change.
            final_prediction = 'No Change'
        else:
            # If a change is detected, use a secondary classifier for the change type.
            final_prediction = classify_greenery_change(
                input_before=before_path, 
                input_after=after_path, 
                threshold_percent=5.0
            )
            
    # Display the results in the console.
    print(f"--- Change Detection Result ---")
    print(f"Model Distance: {distance:.4f}")
    print(f"Decision Threshold: {model_threshold:.4f}")
    
    # Print the final prediction with a corresponding emoji.
    if final_prediction == 'Positive':
        print(f"Prediction: 🌳 {final_prediction}\n")
    elif final_prediction == 'Negative':
        print(f"Prediction: 🪓 {final_prediction}\n")
    else:
        print(f"Prediction: 🔄 {final_prediction}\n")
        
    # Create a plot to visualize the images and the prediction.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f'Prediction: {final_prediction} (Distance: {distance:.4f})', fontsize=14)
    
    # Display the 'Before' image.
    ax1.imshow(Image.open(before_path))
    ax1.set_title("Before")
    ax1.axis('off')
    
    # Display the 'After' image.
    ax2.imshow(Image.open(after_path))
    ax2.set_title("After")
    ax2.axis('off')
    
    # Show the final plot.
    plt.show()
    
    
    
def plot_confusion_matrix_and_metrics(model, data_loader, threshold, device):
    """
    Computes and plots a confusion matrix and classification metrics for a
    Siamese network on a binary change detection task.

    Args:
        model (nn.Module): The trained Siamese network.
        data_loader (DataLoader): The validation data loader.
        threshold (float): The optimal distance threshold to classify change.
        device (torch.device): The device to run the model on.
    """
    # Set the model to evaluation mode for inference.
    model.eval()
    
    all_labels = []
    all_preds = []

    # Generate predictions for the entire dataset.
    with torch.no_grad():
        for before_img, after_img, labels in tqdm(data_loader, desc="Generating Predictions"):
            # Move data to the specified device.
            before_img = before_img.to(device)
            after_img = after_img.to(device)
            
            # Get embeddings from the model.
            output1, output2 = model(before_img, after_img, triplet_bool=False)
            
            # Calculate the pairwise distance between embeddings.
            distances = F.pairwise_distance(output1, output2)
            
            # Make binary predictions based on the optimal threshold.
            preds = (distances >= threshold).long().cpu().numpy()
            
            # Convert the original multi-class labels to binary ground truth labels.
            binary_labels = (labels != 2).long().cpu().numpy()
            
            # Collect the predictions and labels for all batches.
            all_preds.extend(preds)
            all_labels.extend(binary_labels)

    # --- Metrics Calculation ---
    print("\n--- Classification Report ---")
    target_names = ['No Change (Class 0)', 'Change (Class 1)']
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    print(f"F1 Macro Score: {f1_macro:.4f}\n")

    # --- Confusion Matrix Plotting ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()