import pandas as pd
import os

def process_and_average_files(base_filename):
    # Generate file names with appended "-1", "-2", and "-3"
    file_1 = f"{base_filename}-1.csv"
    file_2 = f"{base_filename}-2.csv"

    # Check if all files exist
    if not all(os.path.exists(f) for f in [file_1, file_2]):
        raise FileNotFoundError("One or more files do not exist.")

    # Read the files into DataFrames
    df1 = pd.read_csv(file_1)
    df2 = pd.read_csv(file_2)

    # Make the data have the same length by truncating to the shortest DataFrame
    min_length = min(len(df1), len(df2))
    df1 = df1.iloc[:min_length]
    df2 = df2.iloc[:min_length]

    # Take the average of every row across the three files, considering only numeric columns
    averaged_values = (df1.select_dtypes(include='number').sum(axis=1) +
                       df2.select_dtypes(include='number').sum(axis=1)) / 2
    averaged_df = pd.DataFrame(averaged_values, columns=["Average"])

    # Save the averaged data to a new CSV file
    output_file = f"{base_filename}_averaged.csv"
    averaged_df.to_csv(output_file, index=False)
    print(f"Averaged data saved to {output_file}")

process_and_average_files("10k_dnn_cpu_power")