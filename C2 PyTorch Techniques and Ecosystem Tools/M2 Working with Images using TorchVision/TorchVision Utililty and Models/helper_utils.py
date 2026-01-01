import io
import json
import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from IPython.display import display, HTML
from PIL import Image, ExifTags



def display_images(
    images=None, grid=None, processed_image=None, titles=None, batch=None, predictions=None, labels=None, class_names=None, figsize=(10, 5)
):
    """
    A versatile function to display images in various formats.

    This function can handle several display scenarios:
    - A side-by-side comparison of two images.
    - A grid of images from a tensor.
    - A single processed image tensor.
    - A batch of images with their true and predicted labels.

    Args:
        images (list, optional): A list containing two PIL Images for comparison. Defaults to None.
        grid (torch.Tensor, optional): A tensor representing a grid of images. Defaults to None.
        processed_image (torch.Tensor, optional): A single image tensor to display. Defaults to None.
        titles (tuple, optional): A tuple of strings for the titles of the compared images. Defaults to None.
        batch (torch.Tensor, optional): A batch of images represented as a grid tensor. Defaults to None.
        predictions (torch.Tensor, optional): A tensor of predicted labels for a batch of images. Defaults to None.
        labels (torch.Tensor, optional): A tensor of true labels for a batch of images. Defaults to None.
        class_names (list, optional): A list of strings representing the class names. Defaults to None.
        figsize (tuple, optional): The size of the matplotlib figure. Defaults to (10, 5).
    """

    # Define a helper function to show an image tensor
    def imshow(img_tensor):
        """Utility function to display a tensor as an image."""
        # Unnormalize the image tensor
        img = img_tensor / 2 + 0.5
        # Clip values to be in the valid [0, 1] range for image display
        img = np.clip(img.numpy(), 0, 1)
        # Transpose the tensor from (C, H, W) to (H, W, C) for matplotlib
        plt.imshow(np.transpose(img, (1, 2, 0)))
        # Hide the axes
        plt.axis('off')

    # Check if we need to display images with their predictions and labels
    if predictions is not None and labels is not None and class_names is not None:
        # Ensure the images are provided as a 4D tensor (batch)
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            # Get the number of images in the batch
            num_images = images.size(0)
            # Create subplots for each image
            fig, axes = plt.subplots(1, num_images, figsize=figsize)
            # If there's only one image, wrap the axes object in a list for consistent iteration
            if num_images == 1:
                axes = [axes]
            # Iterate through the axes and display each image with its labels
            for i, ax in enumerate(axes):
                # Set the current axes instance for plotting
                plt.sca(ax)
                # Display the image using the helper function
                imshow(images[i])
                # Get the true and predicted class names
                true_label = class_names[labels[i].item()]
                pred_label = class_names[predictions[i].item()]
                # Set the title for the subplot with true and predicted labels
                ax.set_title(f"True: {true_label}\nPred: {pred_label}")
            # Adjust subplot params for a tight layout
            plt.tight_layout()
            # Show the plot
            plt.show()

    # Check if a batch of images is provided for display
    elif batch is not None:
        # Ensure the batch is a torch tensor
        if isinstance(batch, torch.Tensor):
            # Create a new figure
            plt.figure(figsize=figsize)
            # Display the batch grid
            imshow(batch)
            # Show the plot
            plt.show()

    # Check if a single processed image tensor is provided
    elif processed_image is not None:
        # Ensure the processed image is a torch tensor
        if isinstance(processed_image, torch.Tensor):
            # Create a new figure
            plt.figure(figsize=figsize)
            # Display the image after permuting dimensions from (C, H, W) to (H, W, C)
            plt.imshow(processed_image.permute(1, 2, 0))
            # Hide the axes
            plt.axis('off')
            # Show the plot
            plt.show()

    # Check if a pre-made grid tensor is provided
    elif grid is not None:
        # Ensure the grid is a torch tensor
        if isinstance(grid, torch.Tensor):
            # Create a new figure
            plt.figure(figsize=figsize)
            # Display the grid after permuting dimensions from (C, H, W) to (H, W, C)
            plt.imshow(grid.permute(1, 2, 0))
            # Hide the axes
            plt.axis('off')
            # Show the plot
            plt.show()

    # Check if a list of two PIL images is provided for comparison
    elif images is not None and isinstance(images, list) and len(images) == 2 and all(isinstance(img, Image.Image) for img in images):
        # Create subplots for side-by-side display
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        # Iterate over the axes, images, and titles to display them
        for ax, img, title in zip(axes, images, (titles if titles else ["Image 1", "Image 2"])):
            # Display the image
            ax.imshow(img)
            # Set the title for the subplot
            ax.set_title(title)
            # Hide the axes
            ax.axis('off')
        # Adjust subplot params for a tight layout
        plt.tight_layout()
        # Show the plot
        plt.show()


        
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
    # Initialize the variable to hold the class index data
    imagenet_classes = None

    # Check if the class index file already exists at the specified path
    if os.path.exists(json_path):
        print(f"Loading ImageNet class index from: {json_path}")
        try:
            # Open and load the JSON file
            with open(json_path, 'r') as f:
                imagenet_classes = json.load(f)
            print("Successfully loaded the class index.")
        # Handle cases where the file is not valid JSON
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {json_path}")
            return None
        # Handle other potential exceptions during file reading
        except Exception as e:
            print(f"Error reading {json_path}: {e}")
            return None
    else:
        # If the file does not exist, download it
        print("Downloading ImageNet class index...")
        # URL for the ImageNet class index file
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        try:
            # Send an HTTP GET request to the URL
            response = requests.get(url)
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()
            # Write the downloaded content to the specified local file path
            with open(json_path, 'wb') as f:
                f.write(response.content)
            # Open and load the newly created JSON file
            with open(json_path, 'r') as f:
                imagenet_classes = json.load(f)
            print("Download complete and class index loaded.\n")
        # Handle network-related errors during download
        except requests.exceptions.RequestException as e:
            print(f"Error downloading class index: {e}")
            return None
        # Handle cases where the downloaded content is not valid JSON
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format downloaded from {url}")
            return None
        # Handle any other unexpected errors
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
    # Return the loaded class index dictionary
    return imagenet_classes



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