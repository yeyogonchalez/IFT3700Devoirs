import csv
import numpy as np
import pandas as pd


def cast_to_float_or_nan(lst):
    
    return [float(x) if x.isnumeric() else np.nan for x in lst]


# Open the CSV file and read the contents into a list of rows
with open('D2/filtered_countries_data.csv', 'r') as f:
  reader = csv.reader(f)
  rows = list(reader)

# Get the header row and remove it from the list of rows
header = rows[0]
rows = rows[1:]

# Convert the rows to a NumPy array
data = np.array(rows)

# Create an empty list to store the results
header = header
results=[]

# Iterate over the columns in the data, starting from the second column (index 1)
for col in data.T[1:]:
  # Convert the elements in the column to floats, ignoring empty strings
  # Remove the apostrophes from the values in the column
   
  col=col.astype(float)

  # Calculate the mean, median, maximum, and minimum, ignoring NaN values
  mean = np.nanmean(col)
  median = np.nanmedian(col)
  max_ = np.nanmax(col)
  min_ = np.nanmin(col)


  # Calculate the variance, ignoring NaN values
  variance = np.nanvar(col)
  
  # Count the number of cells with the value 'NA'
  na_count = np.sum(np.isnan(col))
  
  # Append the results for this column to the list
  results.append([mean, median, max_, min_, variance, na_count])
results_format=np.array(results).T.tolist()
test=np.array(results)
print(np.shape(test))
print(len(results_format[0]))
print(len(results))
results_format.insert(0,header)
# Write the results to a CSV file
with open('output.csv', 'w', newline='') as f:
  writer = csv.writer(f)
  writer.writerows(results_format)