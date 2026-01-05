import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from matplotlib.patches import Patch



def optimization_results(results):
    """
    Takes a list of experiment results, sorts them by inference time,
    and displays them in a styled HTML table.

    Args:
        results: A list of dictionaries, where each dictionary
        contains the results of a single experiment.
    """
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(results)
    # Remove duplicate rows based on the 'optimization' column, keeping the last occurrence
    df = df.drop_duplicates(subset=["optimization"], keep="last")

    # Create a copy of the DataFrame for sorting
    df_sorted = df.copy()

    # Handle potential missing data and rename columns for display
    # Fill any missing values in 'peak_mem_mb' with 0
    df_sorted["peak_mem_mb"] = df_sorted["peak_mem_mb"].fillna(0)

    # Rename columns for clarity in the final output
    df_sorted.rename(
        columns={
            "optimization": "Optimization",
            "peak_mem_mb": "Peak Memory (MB)",
            "final_acc": "Final Accuracy (%)",
        },
        inplace=True,
    )

    # Select and reorder columns for the final table
    # Create a new DataFrame with a specific order of columns
    display_df = df_sorted[["Optimization", "Peak Memory (MB)", "Final Accuracy (%)"]]

    # Style and display the table
    # Create a Styler object from the DataFrame and hide the default index
    styler = display_df.style.hide(axis="index")
    # Apply a list of CSS styles to the table
    styler.set_table_styles(
        [
            {"selector": "table", "props": [("width", "100%")]},
            {"selector": "th, td", "props": [("text-align", "left")]},
        ]
    )
    # Apply number formatting to the relevant columns
    styler.format(
        {
            "Peak Memory (MB)": "{:.2f}",
            "Final Accuracy (%)": "{:.2f}",
        }
    )
    # Set the 'white-space' property for all cells to 'normal'
    styler.set_properties(**{"white-space": "normal"})

    # Display the styled table in the output
    display(styler)

    

# Mapping from full names to abbreviated names
name_mapping = {
    "Standard": "STD",
    "Mixed Precision": "MP",
    "Gradient Accumulation (Effective BS: 256-128)": "GA256-128",
    "Gradient Accumulation (Effective BS: 256-64)": "GA256-64",
    "Combined (Effective BS: 256-128)": "Comb256-128",
    "Combined (Effective BS: 256-64)": "Comb256-64",
}



fixed_order = [
    "Standard",
    "Mixed Precision",
    "Gradient Accumulation (Effective BS: 256-128)",
    "Gradient Accumulation (Effective BS: 256-64)",
    "Combined (Effective BS: 256-128)",
    "Combined (Effective BS: 256-64)",
]



def plot_final_accuracy(results):
    """
    Generates and displays a bar chart of final accuracy.

    Args:
        results: A list of dictionaries, where each dictionary
        contains the results of an experiment.
    """
    # Convert the list of results into a pandas DataFrame
    df = pd.DataFrame(results)

    # Remove duplicates by keeping the last occurrence of each optimization technique
    df_unique = df.drop_duplicates(subset=["optimization"], keep="last")

    # Convert the 'optimization' column to a categorical type with a fixed order
    df_unique["optimization"] = pd.Categorical(
        df_unique["optimization"], categories=fixed_order, ordered=True
    )
    # Sort the DataFrame based on the new categorical order
    df_sorted = df_unique.sort_values("optimization")

    # Create abbreviated labels in bold
    x_labels = [name_mapping[opt] for opt in df_sorted["optimization"]]

    # Create a new figure for the plot with a specified size
    plt.figure(figsize=(15, 10))

    # Generate an array of x-positions for the bars
    x_positions = np.arange(len(df_sorted))
    # Create a bar chart with the sorted data
    bars = plt.bar(x_positions, df_sorted["final_acc"], color="#F65B66")

    # Adjust the y-axis limits if the DataFrame is not empty
    if not df_sorted.empty:
        # Determine the maximum accuracy value from the data
        max_acc = df_sorted["final_acc"].max()
        # Set the upper limit of the y-axis
        plt.ylim(top=max(max_acc * 1.1, 100))

    # Set the label for the x-axis
    plt.xlabel("Optimization Technique", fontsize=18)
    # Set the label for the y-axis
    plt.ylabel("Final Accuracy (%)", fontsize=17)
    # Set the title of the plot
    plt.title("Final Accuracy Comparison", fontsize=22, pad=22)

    # Set the tick labels for the x-axis
    plt.xticks(x_positions, x_labels, fontsize=14, weight="bold")
    # Set the font size for the y-axis tick labels
    plt.tick_params(axis="y", labelsize=16)
    # Add labels to the top of each bar
    plt.bar_label(bars, fmt="%.2f%%", fontsize=15)
    # Add a horizontal grid to the plot
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust plot parameters for a tight layout
    plt.tight_layout()
    # Display the plot
    plt.show()

    

def plot_peak_memory(results):
    """
    Generates and displays a bar chart of peak memory usage.

    Args:
        results: A list of dictionaries, where each dictionary
        contains the results of an experiment.
    """
    # Convert the list of results into a pandas DataFrame
    df = pd.DataFrame(results)

    # Remove duplicates by keeping the last occurrence of each optimization technique
    df_unique = df.drop_duplicates(subset=["optimization"], keep="last")

    # Convert the 'optimization' column to a categorical type with a fixed order
    df_unique["optimization"] = pd.Categorical(
        df_unique["optimization"], categories=fixed_order, ordered=True
    )
    # Sort the DataFrame based on the new categorical order
    df_sorted = df_unique.sort_values("optimization")

    # Create abbreviated labels in bold
    x_labels = [name_mapping[opt] for opt in df_sorted["optimization"]]

    # Create a new figure for the plot with a specified size
    plt.figure(figsize=(15, 10))

    # Generate an array of x-positions for the bars
    x_positions = np.arange(len(df_sorted))
    # Create a bar chart with the sorted data
    bars = plt.bar(x_positions, df_sorted["peak_mem_mb"], color="#1C74EB")

    # Get the current y-tick positions
    current_yticks = plt.yticks()[0]
    # Adjust the y-axis limits to provide better spacing
    if len(current_yticks) > 1:
        # Calculate the interval between y-ticks
        interval = current_yticks[1] - current_yticks[0]
        # Get the position of the last y-tick
        last_tick = current_yticks[-1]
        # Calculate a new upper limit for the y-axis
        new_upper_limit = last_tick + (2 * interval) - (interval / 2)
        # Set the upper limit of the y-axis
        plt.ylim(top=new_upper_limit)

    # Set the label for the x-axis
    plt.xlabel("Optimization Technique", fontsize=18)
    # Set the label for the y-axis
    plt.ylabel("Peak Memory (MB)", fontsize=17)
    # Set the title of the plot
    plt.title("Peak Memory Usage Comparison", fontsize=22, pad=22)

    # Set the tick labels for the x-axis
    plt.xticks(x_positions, x_labels, fontsize=14, weight="bold")
    # Set the font size for the y-axis tick labels
    plt.tick_params(axis="y", labelsize=16)
    # Add labels to the top of each bar
    plt.bar_label(bars, fmt="%.0f MB", fontsize=15)
    # Add a horizontal grid to the plot
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust plot parameters for a tight layout
    plt.tight_layout()
    # Display the plot
    plt.show()
