import csv
import numpy as np

# Load the CSV file
def convert(datapath):
    # Open the CSV file in read mode
    csv_file = open(datapath, "r")

    # Create a CSV reader from the file
    csv_reader = csv.reader(csv_file, delimiter=',')

    # Initialize an empty list to store the data
    data = []

    # Read the rows of the CSV file
    for row in csv_reader:
        # Convert each row to a NumPy array and append it to the data list
        data.append(np.array(row))

    # Convert the data list to a NumPy array
    data = np.array(data)

    # Close the CSV file
    csv_file.close()
    return data

def revert(array,output):
    # Write the results to a CSV file
    with open(output, 'w', newline='') as f:
        # Create a CSV writer
        writer = csv.writer(f)

        # Write the rows of the array to the CSV file
        writer.writerows(array)
