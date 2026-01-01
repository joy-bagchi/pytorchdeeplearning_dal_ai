import torchvision.utils as vutils
import matplotlib.pyplot as plt

def display_images(dataloader, figsize=(10, 10), nrow=4):
    """
    Fetches a batch of images from a DataLoader, arranges them in a grid, and displays them.

    Args:
        dataloader (DataLoader): The PyTorch DataLoader to fetch images from.
        figsize (tuple, optional): The size of the figure for display. Defaults to (10, 10).
        nrow (int, optional): Number of images to display in each row of the grid. Defaults to 4.
    """
    # 1. Get one batch of images from the dataloader
    images, _ = next(iter(dataloader))

    # 2. Create a grid from the images
    # normalize=True scales the image pixel values to the range [0, 1]
    grid = vutils.make_grid(images, nrow=nrow, padding=2, normalize=True)

    # 3. Display the grid of images
    plt.figure(figsize=figsize)
    plt.imshow(grid.permute(1, 2, 0)) # Transpose dimensions from (C, H, W) to (H, W, C) for plotting
    plt.axis('off')
    plt.show()