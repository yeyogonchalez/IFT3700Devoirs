import pandas as pd
import statistics
from fractions import Fraction

def intervalMedian(x):
    values=x.split('-')
    print(values)
    int_val=[]
    for val in values:
        if ('/' in val):
            int_val.append(float(Fraction(val)))
        else: int_val.append(float(val))
    median_val=statistics.median(int_val)
    return median_val

# Replace 'FILE_NAME' with the name of the CSV file you want to read
file_name = 'D2/untreated_columns/List_of_countries_by_food_energy_intake_1.csv'

# Read the CSV file into a Pandas DataFrame, specifying that the values are surrounded by single quotes
df = pd.read_csv(file_name)

# Replace the commas with dots in the values
df = df.applymap(lambda x: x.replace(',', '') if isinstance(x, str) else x)

# Remove the apostrophes from the values in the DataFrame
df = df.applymap(lambda x: str(x).strip('"'))

titles=df.iloc[:,1]

# Print the column names and ask the user which ones they want to include in the median calculation
print('Column names:')
for i, column_name in enumerate(df.columns):
    print(f'{i+1}: {column_name}')
selected_columns = input('Enter the column numbers you want to include in the median calculation (separated by commas): ')

# Convert the selected column numbers to a list of integers
selected_columns = [int(x) for x in selected_columns.split(',')]

# Select only the specified columns from the DataFrame
df = df.iloc[:, selected_columns]


# Convert the values to floats, replacing missing values with NaN
#df = df.applymap(lambda x: intervalMedian(str(x)) if '-' in str(x)  else float(x))

# Convert the values to floats, replacing missing values with NaN
df = df.applymap(lambda x: float(x) if x != ('–' or '-') else float('nan'))

# Compute the median value of the selected columns and create a new DataFrame with a single column
median_df = pd.DataFrame({'Median': df.median(axis=1)})

# Replace missing values with the string "NA"
median_df = median_df.applymap(lambda x: "NA" if pd.isnull(x) else x)

# Add the first column (the title of each row) to the new DataFrame
median_df.insert(0, 'Country', 	titles)

# Save the new DataFrame to a CSV file
median_df.to_csv('median.csv', index=False)

print('Median values computed and saved to median.csv')
