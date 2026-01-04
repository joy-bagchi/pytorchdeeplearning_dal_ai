import gc
import json
import os
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm



def run_experiment(experiment_name, experiment_fcn, cases, trainset, device, rerun=False):
    """
    Executes a given experiment function, handling result caching and re-runs.

    Args:
        experiment_name (str): The name of the experiment.
        experiment_fcn (function): The function to be executed for the experiment.
        cases (any): A collection of cases or parameters for the experiment.
        trainset (any): The training dataset.
        device (any): The device to be used for computation.
        rerun (bool): If True, the experiment runs regardless of existing
                      checkpoints, overwriting previous results. Defaults to False.

    Returns:
        dict: The result of the experiment.
    """
    # Define the directory for storing experiment checkpoints
    folder_experiments = "./checkpoint_experiments"
    # Ensure the checkpoint directory exists
    os.makedirs(folder_experiments, exist_ok=True)
    file_path = f"{folder_experiments}/{experiment_name}.json"

    def run_and_save():
        """Executes the experiment, saves the results, and returns them."""
        print(f"Executing experiment '{experiment_name}'...")
        result_experiment = experiment_fcn(cases, trainset, device)
        with open(file_path, "w") as f:
            # Save the experiment results to the file in JSON format
            json.dump(result_experiment, f, indent=4)
        print(f"Results for '{experiment_name}' saved to {file_path}")
        return result_experiment

    # If 'rerun' is True, run the experiment regardless of existing checkpoints
    if rerun:
        print(f"Flag 'rerun' is True. Executing '{experiment_name}' and overwriting any existing results.")
        return run_and_save()

    # If 'rerun' is False, check for a valid checkpoint first
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                result_experiment = json.load(f)
            
            # Preserve original functionality: re-run if results contain infinity
            if any(isinstance(val, float) and val == float("inf") for val in result_experiment.values()):
                print(f"Invalid (infinite) values found in checkpoint for '{experiment_name}'. Re-running.")
                return run_and_save()
            
            # Improved print statement for loading from a valid checkpoint
            print(
                f"Experiment '{experiment_name}' has already been completed. "
                f"Loading results from checkpoint: {file_path}"
            )
            return result_experiment
        except (json.JSONDecodeError, IOError) as e:
            # Handle cases where the file is corrupted or unreadable
            print(f"Could not load checkpoint file {file_path} due to error: {e}. Re-running experiment.")
            return run_and_save()
    else:
        # If no checkpoint exists, run the experiment for the first time
        print(f"Results for experiment '{experiment_name}' not found. Running experiment.")
        return run_and_save()



def download_and_load_cifar10():
    """
    Downloads the CIFAR-10 training dataset and applies transformations.

    Returns:
        The transformed CIFAR-10 training dataset.
    """
    # Define a series of transformations to apply to the images
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),  # The mean for each channel
                (0.5, 0.5, 0.5),  # The standard deviation for each channel
            ),
        ]
    )

    # Specify the local directory for the dataset
    data_path = "./cifar10_data"

    # Check if the dataset directory already exists
    if os.path.exists(data_path) and os.path.isdir(data_path):
        # Set the download flag to False if the directory exists
        download = False
    else:
        # Set the download flag to True if the directory does not exist
        download = True

    # Load the CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,  # Specify the training subset
        download=download,  # Control whether to download the dataset
        transform=transform,  # Apply the defined transformations
    )

    # Return the loaded dataset
    return trainset



def measure_average_epoch_time(loader, device):
    """
    Measures the average time per epoch for data loading and processing.

    Args:
        loader: The data loader to be measured.
        device: The device to which data is moved.

    Returns:
        The average time in seconds for one epoch.
    """
    # A list to store the time taken for each epoch
    epoch_times = []
    # Set the total number of epochs to run
    total_epochs = 5
    # Set the number of initial epochs to be considered as warm-up
    warmup_epochs = 2

    # Iterate over the total number of epochs
    for epoch in tqdm(range(total_epochs), desc="Overall Progress"):
        # Record the start time of the epoch
        start_time = time.time()
        # Iterate over the data in the loader
        for i, data in enumerate(loader, 0):
            # Unpack the data and labels
            inputs, labels = data
            # Move the inputs and labels to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
        # Record the end time of the epoch
        end_time = time.time()

        # Calculate the duration of the current epoch
        current_time = end_time - start_time
        # Add the current epoch's time to the list
        epoch_times.append(current_time)

    # Print a new line for better formatting
    print("")
    # Iterate through the collected epoch times to print them
    for epoch, current_time in enumerate(epoch_times):
        # Create a note if the epoch is a warm-up epoch
        warmup_note = "(warm-up)" if epoch < warmup_epochs else ""
        # Print the time taken for the current epoch
        print(
            f"  Epoch {epoch+1}/{total_epochs} | Time: {current_time:.2f} seconds {warmup_note}"
        )

    # Slice the list to get the times from the measured epochs
    measured_epochs = epoch_times[warmup_epochs:]
    # Calculate the average time of the measured epochs
    average_time = np.mean(measured_epochs)

    # Calculate the number of epochs used for the average
    num_measured_epochs = total_epochs - warmup_epochs
    # Print the final average execution time
    print(
        f"\nAverage execution time (avg of last {num_measured_epochs}): {average_time:.2f} seconds\n"
    )
    # Return the calculated average time
    return average_time



def plot_performance_summary(results_dict, title, xlabel, ylabel="Stable Time per Epoch (ms)"):
    """
    Generates a performance plot with annotations from a dictionary of results.

    Args:
        results_dict: A dictionary where keys are parameters and values are performance metrics.
        title: The title of the plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
    """
    # Create a new dictionary to store only valid results
    valid_times = {k: v for k, v in results_dict.items() if v != float("inf")}

    # Check if there are any valid results to plot
    if not valid_times:
        # Print a message if no valid data is found
        print("No valid results to plot.")
        # Exit the function if there is no data
        return

    # Convert the time values from seconds to milliseconds
    valid_times_ms = {k: v * 1000 for k, v in valid_times.items()}

    # Create a new figure with a specified size
    plt.figure(figsize=(10, 5))

    # Plot the data points with a line and markers
    (line,) = plt.plot(
        list(valid_times_ms.keys()),
        list(valid_times_ms.values()),
        marker="o",
        linestyle="--",
        color="#237B94",
    )

    # Set the title and labels for the plot axes
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Set the x-axis ticks to match the keys in the dictionary
    plt.xticks(list(valid_times_ms.keys()))
    # Add a grid to the plot for better readability
    plt.grid(True)

    # Get the y-values to determine plot limits
    y_values = list(valid_times_ms.values())
    # Set the x-axis limits
    plt.xlim(left=0)
    # Set the y-axis limits with a 10% padding at the top
    plt.ylim(bottom=0, top=max(y_values) + max(y_values) * 0.1)

    # Get the color of the plotted line for annotations
    marker_color = line.get_color()
    # Calculate a vertical offset for the annotations
    vertical_offset = max(y_values) * 0.06

    # Iterate over the data points to add annotations
    for x, y in valid_times_ms.items():
        # Add a text annotation at each data point
        plt.text(
            x,
            y + vertical_offset,
            f"{y:.2f}",
            ha="center",
            color=marker_color,
        )

    # Display the plot
    plt.show()



def visualize_dataloader_efficiency(loader_dict, device):
    """
    Analyzes and visualizes the efficiency of multiple data loaders.

    Args:
        loader_dict: A dictionary of data loaders to be analyzed.
        device: The device for data transfer.
    """
    # Define the number of batches to analyze
    num_batches = 52
    # Define the number of initial batches to be used for warm-up
    warmup_batches = 2

    # A list to store labels for the plot
    labels = []
    # A list to store average work times
    avg_work_times = []
    # A list to store average wait times
    avg_wait_times = []

    # Iterate through the dictionary of data loaders
    for label, loader in loader_dict.items():
        # Lists to store individual wait and work times
        wait_times, work_times = [], []
        # Initialize the end time of the previous work period
        end_of_previous_work = time.time()

        # Begin a try-except block to handle potential errors
        try:
            # Create an iterator for the data loader
            data_iter = iter(loader)
            # Loop for the specified number of batches
            for batch_idx in range(num_batches):
                # Get the next batch of data
                data = next(data_iter)
                # Record the start time of the current work period
                start_of_current_work = time.time()
                # Calculate and append the wait time
                wait_times.append(start_of_current_work - end_of_previous_work)

                # Unpack the data and labels from the batch
                inputs, labels_tensor = data
                # Move the inputs to the specified device
                inputs.to(device)
                # Move the labels to the specified device
                labels_tensor.to(device)

                # Record the end time of the current work period
                end_of_current_work = time.time()
                # Calculate and append the work time
                work_times.append(end_of_current_work - start_of_current_work)
                # Update the end time for the next iteration
                end_of_previous_work = end_of_current_work

            # Check if there are enough work times after warm-up
            if len(work_times) > warmup_batches:
                # Calculate the average work time, excluding warm-up batches
                avg_work = np.mean(work_times[warmup_batches:])
                # Calculate the average wait time, excluding warm-up batches
                avg_wait = np.mean(wait_times[warmup_batches:])

                # Append the label to the list
                labels.append(label)
                # Append the average work time
                avg_work_times.append(avg_work)
                # Append the average wait time
                avg_wait_times.append(avg_wait)
            # Handle cases where there are not enough batches
            else:
                # Print a message indicating the reason for skipping
                print(f"--> Skipping '{label}' due to insufficient batches.")

        # Catch the StopIteration exception if the dataset is exhausted
        except StopIteration:
            # Print a warning message
            print(
                f"Warning: Dataset exhausted for '{label}' before completing {num_batches} batches."
            )
        # Catch a generic RuntimeError
        except RuntimeError as e:
            # Print an error message
            print(f"\n❌ ERROR with {label}: {e}")

        # Ensure resources are cleaned up regardless of errors
        finally:
            # Delete the loader object
            del loader
            # Force garbage collection
            gc.collect()
            # If a CUDA device is available, clear its cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Check if any valid results were collected
    if not labels:
        # Print a message if there are no results to plot
        print("\nNo valid loader results to plot.")
        # Exit the function
        return

    # --- Plot Generation ---
    # Lists to store work and wait times as percentages
    work_percentages = []
    wait_percentages = []
    # Calculate the raw total time for each loader
    total_times_raw = [w + a for w, a in zip(avg_work_times, avg_wait_times)]

    # Loop through the average work and wait times
    for work, wait in zip(avg_work_times, avg_wait_times):
        # Calculate the total time for the current loader
        total_time = work + wait
        # Check if the total time is positive
        if total_time > 0:
            # Calculate and append the work percentage
            work_percentages.append((work / total_time) * 100)
            # Calculate and append the wait percentage
            wait_percentages.append((wait / total_time) * 100)
        else:
            # Append zero if the total time is not positive
            work_percentages.append(0)
            wait_percentages.append(0)

    # Create a figure with a specified size
    fig = plt.figure(figsize=(10, 8))
    # Create a grid for the subplots
    gs = gridspec.GridSpec(1, 2, width_ratios=[8, 2])

    # Add a subplot to the figure grid
    ax = fig.add_subplot(gs[0, 0])
    # Create an array of indices for the x-axis
    indices = np.arange(len(labels))

    # Plot the work percentages as a bar chart
    p1 = ax.bar(indices, work_percentages, color="#1C74EB", label="GPU Active Time (%)")
    # Plot the wait percentages as a stacked bar chart
    p2 = ax.bar(
        indices,
        wait_percentages,
        bottom=work_percentages,
        color="#FAB901",
        label="GPU Idle / Waiting Time (%)",
    )

    # Set the y-axis limits
    ax.set_ylim(bottom=0, top=100)

    # Determine a dynamic font size for the annotations
    dynamic_fontsize = max(8, 20 - len(labels))
    # Loop through the bars to add text annotations
    for i in range(len(labels)):
        # Check if the work percentage is greater than zero
        if work_percentages[i] > 0:
            # Add a text annotation with the work percentage
            ax.text(
                i,
                work_percentages[i] + 0.5,
                f"{work_percentages[i]:.1f}%",
                ha="center",
                va="bottom",
                color=p1[i].get_facecolor(),
                fontsize=dynamic_fontsize,
                weight="bold",
            )

    # Set the label for the y-axis
    ax.set_ylabel("Percentage of Average Time per Batch (%)")
    # Set a title for the entire figure
    fig.suptitle(
        "DataLoader Performance Comparison (Efficiency)", fontsize=16, weight="bold"
    )
    # Set the x-axis ticks
    ax.set_xticks(indices)
    # Set the x-axis tick labels and rotate them
    ax.set_xticklabels(labels, rotation=45, ha="right")
    # Add a grid to the plot
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust the plot layout
    plt.tight_layout()
    # Display the plot
    plt.show()
