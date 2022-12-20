import os
import csv


def ensure_same_size(lists):
  # Get the size of the first list
  size = len(lists[0])
  
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
counter=0
# Iterate through the csv files in the folder
for csv_file in csv_files:
    countries=ensure_same_size(countries)
    counter+=1
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        # Read the rows into a list of tuples
        rows = list(reader)
        # Iterate through the list of tuples
        
        for row in rows:
            altered=False
            
            #print(row)
            country, category = row
            # Special Cases
            if 'Anguilla' in country:country='Anguilla' ; altered=True
            if 'Puerto Rico' in country:country='Puerto Rico'  ; altered=True
            if 'Aruba' in country:country='Aruba'  ; altered=True
            if 'Bolivia' in country:country='Bolivia'  ; altered=True
            if 'British Virgin Islands' in country:country='British Virgin Islands'; altered=True
            if 'Cayman Islands' in country:country='Cayman Islands'    ; altered=True
            if 'Cook Islands' in country:country='Cook Islands'; altered=True
            if 'Czech' in country:country='Czech Republic'; altered=True
            if 'Denmark (including Faroe Islands)' in country:country='Denmark'; altered=True
            if 'Eswatini' in country:country='Eswatini'; altered=True
            if 'French Polynesia' in country:country='French Polynesia'; altered=True
            if "Côte d'Ivoire" in country:country="Côte d'Ivoire"; altered=True
            if 'Faroe Islands (Denmark)' in country:country="Faroe Islands"; altered=True
            if 'Gibraltar' in country:country="Gibraltar"; altered=True
            if 'Guadeloupe' in country:country="Guadeloupe"; altered=True
            if 'Guam' in country:country="Guam"; altered=True
            if 'Hong Kong' in country:country="Hong Kong"; altered=True
            if 'Iran' in country:country="Iran"; altered=True
            if 'Iraq (excluding Kurdistan)' in country:country="Iraq"; altered=True
            if 'Isle of Man' in country:country="Isle of Man"; altered=True
            if 'Lao' in country:country="Laos"; altered=True
            if 'Macau' in country:country="Macau"; altered=True
            if 'Martinique' in country:country="Martinique"; altered=True
            if 'Mayotte' in country:country="Mayotte"; altered=True
            if 'Micronesia' in country:country="Micronesia"; altered=True
            if 'Montserrat' in country:country="Montserrat"; altered=True
            if 'Myanmar' in country:country="Myanmar"; altered=True
            if 'New Caledonia' in country:country="New Caledonia"; altered=True
            if 'Northern Mariana Islands' in country:country="Northern Mariana Islands"; altered=True
            if 'Palestine' in country:country="Palestine"; altered=True
            if 'Pitcairn Islands' in country:country="Pitcairn Islands"; altered=True
            if 'Réunion' in country:country="Réunion"; altered=True
            if 'Russia' in country:country="Russia"; altered=True
            if 'Saint Helena' in country:country="Saint Helena"; altered=True
            if 'Saint Martin' in country:country="Saint Martin" ; altered=True
            if 'Saint Pierre and Miquelon' in country:country="Saint Pierre and Miquelon"; altered=True
            if 'São Tomé and Príncipe' in country:country="Sao Tome and Principe"; altered=True
            if 'Sint Maarten' in country:country="Sint Maarten"; altered=True
            if 'Tokelau' in country:country="Tokelau"; altered=True
            if 'Trinidad' in country:country="Trinidad"; altered=True
            if 'Turks and Caicos Islands' in country:country="Turks and Caicos Islands"; altered=True
            if 'Great Britain' in country:country="United Kingdom"; altered=True
            if 'United States of America' in country:country="United States"; altered=True
            if 'Venezuela' in country:country="Venezuela"; altered=True
            if 'U.S.' in country:country="Virgin Islands"; altered=True
            if 'Wallis and Futuna' in country:country="Wallis and Futuna"; altered=True
            if 'Western Sahara' in country:country="Western Sahara"; altered=True
            if 'Curacao' in country:country="Curaçao"; altered=True
            if 'Republic of Congo' in country:country="Congo"; altered=True
            if 'DR' in country:country="Congo"; altered=True
            # Check if the country is already in the list
            #if altered:category+='alt'
            found = False
            for i, c in enumerate(countries):
                if c[0] == country:
                    # If the country is already in the list, append the category
                    countries[i].append(category)
                    found = True
                    print(countries[i])
                    print(counter)
                    if countries[i][counter]!=category:
                        countries[i][counter]=category
                        countries[i]=countries[i][0:counter]
                    break
            if not found:
                #print(len(countries))
                # If the country is not in the list, create a new list for it
                countries.append([country] + ["NA"] * (len(countries[0]) - 1))
                countries[-1][-1] = category

 

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
# The file has now been saved with the name 'data.csv'

