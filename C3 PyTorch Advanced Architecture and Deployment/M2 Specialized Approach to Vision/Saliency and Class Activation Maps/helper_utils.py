import os
import ipywidgets as widgets
from IPython.display import display, clear_output

def plot_widget(compute_gradcam, visualize_gradcam, model, transform, device, folder="images"):
    """
    Displays a widget to select and visualize GradCAM for images in a folder.

    Args:
        compute_gradcam: function(img_path, model, transform, device) -> (img_display, heatmap, pred_class, pred_score)
        visualize_gradcam: function(img, heatmap, pred_class, pred_score, title)
        model: PyTorch model with weights loaded.
        transform: Torch transform for the input image.
        device: torch.device
        folder: str, folder to look in (default: "images")
    """
    def get_jpg_files(folder=folder):
        """
        Returns a sorted list of .jpg files (case-insensitive) in the chosen folder.
        """
        if not os.path.exists(folder):
            return []
        return sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])

    out = widgets.Output()

    def gradcam_widget_view(image_name):
        """
        Widget callback: runs GradCAM and visualization for the selected image.
        All plots and outputs are displayed in the widget output area.
        """
        img_path = os.path.join(folder, image_name)
        with out:
            clear_output(wait=True)
            print(f"Showing GradCAM for: {image_name}")
            img_display, heatmap, pred_class, pred_score = compute_gradcam(
                img_path, model, transform, device
            )
            if img_display is not None:
                title = os.path.splitext(image_name)[0]
                visualize_gradcam(img_display, heatmap, pred_class, pred_score, title)
            else:
                print(f"Could not process {image_name}.")

    jpg_list = get_jpg_files(folder)
    if not jpg_list:
        display(widgets.HTML(
            value=f"<b style='color: red;'>No .jpg files found in the <code>{folder}/</code> folder. Please add images and rerun.</b>"
        ))
    else:
        dropdown = widgets.Dropdown(
            options=jpg_list,
            description='Select image:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%'),  # Wider, easier to use
            disabled=False
        )
        display(widgets.HTML(f"<h3>GradCAM Visualizer</h3>Select an image from <code>{folder}</code>:"))
        widgets.interact(gradcam_widget_view, image_name=dropdown)
        display(out)