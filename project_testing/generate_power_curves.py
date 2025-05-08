import os
import glob
import pandas as pd

import matplotlib.pyplot as plt

# Directory containing the CSV files
csv_dir = "/home/sschott/CSCI582-Final-Project/pinns-torch/project_testing"

# File pattern to match
file_pattern = os.path.join(csv_dir, "*_*_*_power_averaged.csv")

# Initialize the plot
plt.figure(figsize=(10, 6))

# Iterate over all matching files
for file_path in glob.glob(file_pattern):
    # Extract num_data, app, and hw from the filename
    filename = os.path.basename(file_path)
    parts = filename.split("_")
    if len(parts) < 3:
        continue  # Skip files that don't match the expected pattern
    num_data = parts[0].replace("k", "")
    app = parts[1]
    hw = parts[2]

    # Read the CSV file, ignoring the header
    data = pd.read_csv(file_path, skiprows=1, header=None)

    # Plot the data
    plt.plot(data[0], label=f"{num_data}000 datapoints using {app} on {hw}")

# Add legend, labels, and title
plt.legend()
plt.xlabel("Data Point Index")
plt.ylabel("Power (mW)")
plt.title("Power Consumption Across Different Configurations")
plt.grid(True)

# Show the plot
plt.tight_layout()
output_path = os.path.join(csv_dir, "power_curves_plot.png")
plt.savefig(output_path)