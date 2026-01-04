import lightning.pytorch as pl
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML, display
from lightning.pytorch import LightningDataModule, LightningModule



def run_full_training(model, data_module, epochs=5):
    """
    Conducts a full training run and returns the final metrics.

    Args:
        model (LightningModule): The LightningModule to be trained.
        data_module (LightningDataModule): The LightningDataModule providing the data.
        epochs (int, optional): The number of epochs to train for. Defaults to 5.

    Returns:
        dict: A dictionary containing the final train_loss, train_accuracy,
              val_loss, and val_accuracy.
    """
    print(f"--- Running Training For {epochs} Epochs ---")

    # Initialize a Lightning Trainer with specified configurations
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=False,
    )

    # Begin the model training process
    trainer.fit(model, data_module)

    # Extract final metrics from the trainer after fitting
    final_metrics = {
        "train_accuracy": round(
            trainer.callback_metrics["train_accuracy"].item() * 100, 2
        ),
        "val_accuracy": round(trainer.callback_metrics["val_accuracy"].item() * 100, 2),
    }

    # Return the dictionary of final training and validation metrics
    return final_metrics



def display_profiler_logs(profiler_object, head=10):
    """
    Processes and displays a summary of profiler event data.

    Args:
        profiler_object: The profiler instance that contains the recorded
                         performance data.
        head: An integer specifying the number of top time-consuming
              operations to display in the output table.
    """
    # Retrieve the aggregated averages of all function events from the profiler.
    profiler_events = profiler_object.function_events.key_averages()

    # Initialize a list to store dictionaries of processed event data.
    event_list = []
    # Iterate through each event to extract and format the required metrics.
    for event in profiler_events:
        # Get the total CUDA time and convert it from microseconds to milliseconds.
        cuda_time_total_ms = getattr(event, "cuda_time_total", 0) / 1000
        # Get the total CPU time and convert it from microseconds to milliseconds.
        cpu_time_total_ms = event.cpu_time_total / 1000
        # Append a dictionary with the summarized data for the current event.
        event_list.append(
            {
                "Action": event.key,
                "Total Time (ms)": cpu_time_total_ms + cuda_time_total_ms,
                "Calls": event.count,
            }
        )
    # Convert the list of event data into a pandas DataFrame.
    full_df = pd.DataFrame(event_list)

    # Sort the DataFrame by total time in descending order to identify the
    # most expensive operations.
    full_df_sorted = full_df.sort_values(by="Total Time (ms)", ascending=False)

    # Select the top N rows based on the 'head' parameter for the final output.
    display_df = full_df_sorted.head(head).copy()
    # The following steps format the DataFrame for better readability.
    # Reset the DataFrame index to use it for sequencing.
    display_df.reset_index(inplace=True)
    # Rename the old index column to reflect its meaning.
    display_df.rename(columns={"index": "Operation Sequence"}, inplace=True)
    # Add a new column for user-friendly row numbering starting from 1.
    display_df["Row"] = display_df.index + 1
    # Reorder the columns for a logical presentation.
    display_df = display_df[
        ["Row", "Operation Sequence", "Action", "Total Time (ms)", "Calls"]
    ]

    # Use the pandas Styler to apply custom CSS for a clean HTML table.
    # Hide the DataFrame's own index as a 'Row' column is now used.
    styler = display_df.style.hide(axis="index")
    # Apply table-wide styles, such as setting width and text alignment.
    styler.set_table_styles(
        [
            {"selector": "table", "props": [("width", "100%")]},
            {"selector": "th, td", "props": [("text-align", "left")]},
        ]
    # Set properties for cell content, such as allowing text to wrap.
    ).set_properties(**{"white-space": "normal"})

    # Render the final styled table in the output.
    display(styler)

    

def display_model_computation_logs(profiler_object):
    """
    Analyzes profiler data and displays a styled table showing overall time
    plus the top 4 model computation kernels.

    Args:
        profiler_object: The profiler object containing event data.
    """
    # Create the comprehensive DataFrame from the profiler events
    profiler_events = profiler_object.function_events.key_averages()
    event_list = []
    # Iterate through profiler events to extract key metrics
    for event in profiler_events:
        # Get the total CUDA time and convert to milliseconds
        cuda_time_total_ms = getattr(event, "cuda_time_total", 0) / 1000
        # Get the total CPU time and convert to milliseconds
        cpu_time_total_ms = event.cpu_time_total / 1000
        # Append a dictionary of metrics for the current event to a list
        event_list.append(
            {
                "Action": event.key,
                "Total Time (ms)": cpu_time_total_ms + cuda_time_total_ms,
                "Calls": event.count,
            }
        )
    # Create a DataFrame from the list of event dictionaries
    full_df = pd.DataFrame(event_list)

    # Sort all events by time to get the correct Operation Sequence
    # Sort the DataFrame by the 'Total Time (ms)' column in descending order
    full_df_sorted = full_df.sort_values(by="Total Time (ms)", ascending=False)
    # Reset the DataFrame's index to a default integer index
    full_df_sorted.reset_index(inplace=True)
    # Rename the 'index' column to 'Operation Sequence'
    full_df_sorted.rename(columns={"index": "Operation Sequence"}, inplace=True)

    # Isolate the 'ProfilerStep*' and the key computational operations
    # Select the row where the 'Action' is 'ProfilerStep*'
    profiler_step_row = full_df_sorted[
        full_df_sorted["Action"] == "ProfilerStep*"
    ].head(1)

    # Define the pattern for the computational kernels to be included
    target_ops_pattern = (
        "aten::conv2d|aten::addmm|aten::linear|convolution_backward|AddmmBackward"
    )

    # Exclude the 'ProfilerStep*' row before searching for computation kernels
    # Select all rows that are not 'ProfilerStep*'
    other_rows = full_df_sorted[full_df_sorted["Action"] != "ProfilerStep*"]
    # Filter for rows whose 'Action' matches the specified pattern
    computation_rows = other_rows[other_rows["Action"].str.contains(target_ops_pattern)]

    # Get the top 4 most time-consuming computational rows
    # Select the first four rows from the filtered DataFrame
    top_4_computation = computation_rows.head(4)

    # Combine and prepare the final DataFrame for display
    # Concatenate the 'ProfilerStep*' row and the top 4 computation rows
    display_df = pd.concat([profiler_step_row, top_4_computation]).copy()
    # Add a new 'Row' column with a sequential integer for each row
    display_df["Row"] = range(1, len(display_df) + 1)
    # Reorder the columns for final display
    display_df = display_df[
        ["Row", "Operation Sequence", "Action", "Total Time (ms)", "Calls"]
    ]

    # Style and display the final table
    # Create a Styler object from the DataFrame
    styler = display_df.style.hide(axis="index")
    # Apply custom table styles
    styler.set_table_styles(
        [
            {"selector": "table", "props": [("width", "100%")]},
            {"selector": "th, td", "props": [("text-align", "left")]},
        ]
    ).set_properties(**{"white-space": "normal"})

    # Display the styled table in the output
    display(styler)


    
def _create_profiler_df(profiler_object):
    """
    Converts a profiler object to a DataFrame for analysis.

    Args:
        profiler_object: The profiler object containing event data.

    Returns:
        DataFrame: A pandas DataFrame summarizing profiler events.
    """
    # Retrieve key averaged events from the profiler object
    events = profiler_object.function_events.key_averages()
    # Create a list of dictionaries from the events
    event_list = [
        {
            "Action": event.key,
            "Total Time (ms)": (
                getattr(event, "cuda_time_total", 0) + event.cpu_time_total
            )
            / 1000,
        }
        for event in events
    ]
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(event_list)
    # Group the DataFrame by 'Action' and calculate the maximum time for each
    return df.groupby("Action", as_index=False)["Total Time (ms)"].max()



def display_comparison_report(profiler_before, profiler_after):
    """
    Displays a styled comparison table for the total time and top 4
    computational kernels from two profiler runs.

    Args:
        profiler_before: The profiler object for the initial run.
        profiler_after: The profiler object for the subsequent run.
    """
    # Create DataFrames for both profiler runs
    df_before = _create_profiler_df(profiler_before)
    df_after = _create_profiler_df(profiler_after)

    # Identify the top 4 computational kernels from the 'before' run
    # Define the pattern for the computational kernels to be included
    target_ops_pattern = (
        "aten::conv2d|aten::addmm|aten::linear|convolution_backward|AddmmBackward"
    )

    # Filter for computational rows in the 'before' DataFrame
    comp_rows_before = df_before[df_before["Action"].str.contains(target_ops_pattern)]
    # Sort the filtered rows by time and select the top 4
    top_4_kernels = comp_rows_before.sort_values(
        by="Total Time (ms)", ascending=False
    ).head(4)

    # Define the exact list of operations to compare
    # Create a list of operation names, including 'ProfilerStep*' and the top 4 kernels
    op_names = ["ProfilerStep*"] + top_4_kernels["Action"].tolist()

    # Build the comparison data list
    # Set the 'Action' column as the index for both DataFrames
    df_before.set_index("Action", inplace=True)
    df_after.set_index("Action", inplace=True)

    # Iterate through the list of operation names to build comparison data
    comparison_data = []
    for op_name in op_names:
        # Get the total time for the 'before' run
        time_before = df_before.loc[op_name]["Total Time (ms)"]
        # Get the total time for the 'after' run, handling cases where the operation may be missing
        time_after = (
            df_after.loc[op_name]["Total Time (ms)"]
            if op_name in df_after.index
            else 0.0
        )

        # Append a dictionary with the comparison data
        comparison_data.append(
            {
                "Operation": op_name,
                "Total Time Before (ms)": time_before,
                "Total Time After (ms)": time_after,
            }
        )

    # Create and style the final DataFrame
    # Convert the list of dictionaries to a DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # Create a Styler object for the DataFrame
    styler = comparison_df.style.hide(axis="index")
    # Apply custom styles to the table
    styler.set_table_styles(
        [
            {"selector": "table", "props": [("width", "100%")]},
            {"selector": "th", "props": [("text-align", "left")]},
            {"selector": "td", "props": [("text-align", "left"), ("padding", "8px")]},
        ]
    ).set_properties(**{"white-space": "normal"})

    # Display the styled table in the output
    display(styler)

    

def display_metrics_comparison(baseline_results, efficient_results):
    """
    Displays a styled comparison table for the final metrics of two model runs.

    Args:
        baseline_results: A dictionary of metrics for the baseline model.
        efficient_results: A dictionary of metrics for the efficient model.
    """
    # Structure the data for the DataFrame
    # Create a list of dictionaries to hold the comparison data
    data = [
        {
            "Metric": "Training Accuracy (%)",
            "Baseline Model": baseline_results["train_accuracy"],
            "Efficient Model": efficient_results["train_accuracy"],
        },
        {
            "Metric": "Validation Accuracy (%)",
            "Baseline Model": baseline_results["val_accuracy"],
            "Efficient Model": efficient_results["val_accuracy"],
        },
    ]

    # Create the DataFrame
    # Convert the list of dictionaries into a pandas DataFrame
    comparison_df = pd.DataFrame(data)

    # Style the DataFrame for clear presentation
    # Create a Styler object and hide the index
    styler = comparison_df.style.hide(axis="index")
    # Apply custom table styles
    styler.set_table_styles(
        [
            {"selector": "table", "props": [("width", "100%")]},
            {"selector": "th", "props": [("text-align", "left")]},
            {"selector": "td", "props": [("text-align", "left"), ("padding", "8px")]},
        ]
    )
    # Format the numeric columns to two decimal places
    styler.format({"Baseline Model": "{:.2f}", "Efficient Model": "{:.2f}"})

    # Display the styled table in the output
    display(styler)
