import csv

# Set the input and output file names
input_file = 'input_file'
output_file = 'filtered_data.csv'

# Open the input file and read the rows
with open(input_file, 'r') as f_in:
    reader = csv.reader(f_in)
    rows = [row for row in reader]

# Filter the rows and keep only those with fewer than 12 cells with the value "NA"
filtered_rows = [row for row in rows if row.count('NA') <= 12]
[print(True) for row in filtered_rows if row.count('NA') < 12]
# Open the output file and write the filtered rows
with open(output_file, 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerows(filtered_rows)
