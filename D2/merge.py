import os
import csv


def ensure_same_size(lists):
  # Get the size of the first list
  size = len(lists[0])-1
  
  # Use map to apply a function to each list in the input
  # The function will check the size of the list and append "NA" to the end if necessary
  lists = list(map(lambda l: l + ["NA"] * (size - len(l)), lists))
  
  return lists

# Set the path to the folder containing the CSV files
folder_path = "D2/treated columns"

# Get a list of all the files in the folder
files = os.listdir(folder_path)

# Filter the list of files to only include CSV files
csv_files = ['D2/treated columns/' + f for f in files if f.endswith('.csv')]

# csv_files now contains a list of the CSV files in the folder

countries = [['Country'],['United Kingdom'],['France']]

# Iterate through the csv files in the folder
for csv_file in csv_files:
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        # Read the rows into a list of tuples
        rows = list(reader)
        # Iterate through the list of tuples
        counter=0
        for row in rows:
            
            #print(row)
            country, category = row
            # Special Cases
            if 'Anguilla' in country:country='Anguilla'
            if 'Puerto Rico' in country:country='Puerto Rico'  
            if 'Aruba' in country:country='Aruba'  
            if 'Bolivia' in country:country='Bolivia'  
            if 'British Virgin Islands' in country:country='British Virgin Islands'
            if 'Cayman Islands' in country:country='Cayman Islands'    
            if 'Cook Islands' in country:country='Cook Islands'
            if 'Czech' in country:country='Czech Republic'
            if 'Denmark (including Faroe Islands)' in country:country='Denmark'
            if 'Eswatini' in country:country='Eswatini'
            if 'French Polynesia' in country:country='French Polynesia'
            if "Côte d'Ivoire" in country:country="Côte d'Ivoire"
            if 'Faroe Islands (Denmark)' in country:country="Faroe Islands"
            if 'Gibraltar' in country:country="Gibraltar"
            if 'Guadeloupe' in country:country="Guadeloupe"
            if 'Guam' in country:country="Guam"
            if 'Hong Kong' in country:country="Hong Kong"
            if 'Iran' in country:country="Iran"
            if 'Iraq (excluding Kurdistan)' in country:country="Iraq"
            if 'Isle of Man' in country:country="Isle of Man"
            if 'Lao' in country:country="Laos"
            if 'Macau' in country:country="Macau"
            if 'Martinique' in country:country="Martinique"
            if 'Mayotte' in country:country="Mayotte"
            if 'Micronesia' in country:country="Micronesia"
            if 'Montserrat' in country:country="Montserrat"
            if 'Myanmar' in country:country="Myanmar"
            if 'New Caledonia' in country:country="New Caledonia"
            if 'Northern Mariana Islands' in country:country="Northern Mariana Islands"
            if 'Palestine' in country:country="Palestine"
            if 'Pitcairn Islands' in country:country="Pitcairn Islands"
            if 'Réunion' in country:country="Réunion"
            if 'Russia' in country:country="Russia"
            if 'Saint Helena' in country:country="Saint Helena"
            if 'Saint Martin' in country:country="Saint Martin"
            if 'Saint Pierre and Miquelon' in country:country="Saint Pierre and Miquelon"
            if 'São Tomé and Príncipe' in country:country="Sao Tome and Principe"
            if 'Sint Maarten' in country:country="Sint Maarten"
            if 'Tokelau' in country:country="Tokelau"
            if 'Trinidad' in country:country="Trinidad"
            if 'Turks and Caicos Islands' in country:country="Turks and Caicos Islands"
            if 'Great Britain' in country:country="United Kingdom"
            if 'United States of America' in country:country="United States"
            if 'Venezuela' in country:country="Venezuela"
            if 'U.S.' in country:country="Virgin Islands"
            if 'Wallis and Futuna' in country:country="Wallis and Futuna"
            if 'Western Sahara' in country:country="Western Sahara"
            if 'Curacao' in country:country="Curaçao"
            if 'Republic of Congo' in country:country="Congo"
            if 'DR' in country:country="Congo"
            # Check if the country is already in the list
            found = False
            for i, c in enumerate(countries):
                if c[0] == country:
                    # If the country is already in the list, append the category
                    countries[i].append(category)
                    found = True
                    break
            if not found:
                #print(len(countries))
                # If the country is not in the list, create a new list for it
                countries.append([country] + ["NA"] * (len(countries[0]) - 1))
                countries[-1][-1] = category
            
        ensure_same_size(countries)

# Fill any remaining gaps in the lists with the value "NA"
for c in countries:
    while len(c) < len(countries[0]):
        c.append("NA")

# countries is now a list of lists where each list contains the country name followed by the categories


# Open a file for writing
with open("data.csv", "w", newline="") as file:
  # Create a CSV writer
  writer = csv.writer(file)
  same_size=True
  # Write each row of the list to the CSV file
  for row in countries:
    #print(len(row))
    if len(row)!=41:
        print(row)
        same_size=False
    writer.writerow(row)
    print(same_size)
# The file has now been saved with the name 'data.csv'

