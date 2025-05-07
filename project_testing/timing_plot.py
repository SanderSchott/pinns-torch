import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Function to convert time strings (e.g., "1:43.620") to seconds
def time_to_seconds(time_str):
    minutes, seconds = map(float, time_str.split(':'))
    return minutes * 60 + seconds

# Read the CSV file
file_path = './timing_results.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Convert runtime columns to seconds and calculate the average runtime
for trial in ['Trial_1', 'Trial_2', 'Trial_3']:
    df[trial] = df[trial].apply(time_to_seconds)
df['Average_Runtime'] = df[['Trial_1', 'Trial_2', 'Trial_3']].mean(axis=1)

# Create a bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Group data by Datapoints and Application
grouped = df.groupby(['Datapoints', 'Application'])

# Prepare data for plotting
datapoints = sorted(df['Datapoints'].unique(), key=lambda x: int(x[:-1]))  # Sort by numeric value
applications = df['Application'].unique()
colors = plt.cm.tab10(range(len(applications)))  # Generate distinct colors for applications

bar_width = 0.2
x_indices = range(len(datapoints))

for i, app in enumerate(applications):
    avg_runtimes = [
        grouped.get_group((dp, app))['Average_Runtime'].values[0] if (dp, app) in grouped.groups else 0
        for dp in datapoints
    ]
    x_positions = [x + i * bar_width for x in x_indices]
    ax.bar(x_positions, avg_runtimes, bar_width, label=app, color=colors[i])

# Customize the plot
ax.set_title('Average Runtimes by Application and Datapoints')
ax.set_xlabel('Datapoints')
ax.set_ylabel('Average Runtime (seconds)')
ax.set_xticks([x + (len(applications) - 1) * bar_width / 2 for x in x_indices])
ax.set_xticklabels(datapoints)
ax.legend(title='Application')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x // 60)}:{int(x % 60):02d}'))

# Show the plot
plt.tight_layout()
plt.show()