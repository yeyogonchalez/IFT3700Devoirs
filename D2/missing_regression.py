import numpy as np
import csv_to_np as ctn # support script that converts csv to np array
from sklearn.linear_model import ElasticNet
import json

countries_data=ctn.convert('D2/filtered_countries_data.csv')

statistics=ctn.convert('D2/statistics.csv')

num_regressions=2

# Get the header row and remove it from the list of rows
header = countries_data[0]
countries_data = countries_data[1:,:]
statistics=statistics[1:,:]


# used as a separate to variable to store the predictictions separately
modified_countries=countries_data.copy() 

for i in range(1,countries_data.shape[1]):
  column = countries_data[:, i]
  median = statistics[1,i]

  modified_countries[:,i] = np.where(column=='NA', median, column)


# Linear Regressions
for i in range(num_regressions):
    for j in range(1,countries_data.shape[1]):
        predictions=[]
        y = modified_countries[:, j]
        y=y.astype(float)

        X = np.delete(modified_countries, j, axis=1)

        X=X[:,1:]

        X=X.astype(float)

        indices = np.where(countries_data[:, j] == 'NA')[0]

        y_train = np.delete(y, indices, axis=0)
        X_train = np.delete(X, indices, axis=0)

        X_pred = X[indices]

        lr=ElasticNet()

        lr.fit(X,y)
        
        for index in indices:
            prediction=lr.predict(X[index].reshape(1, -1))
            modified_countries[index][j]=prediction[0]


# Duplicate columns
    
duplicated_countries = np.empty((countries_data.shape[0], countries_data.shape[1]*2),dtype=object)

for i in range(1,countries_data.shape[1]):
    # assign the values of the original column to the first half of the new column
    duplicated_countries[:, i*2] = modified_countries[:, i]
    median = float(statistics[1,i])
    
    # assign the values of the original column to the second half of the new column
    for j in range(countries_data.shape[0]):
        if float(duplicated_countries[j, i*2])<=median:
            duplicated_countries[j,i*2+1]=0
        else:
            duplicated_countries[j,i*2+1]=1
    

    
    
duplicated_countries[:,1] =  modified_countries[:,0]
#duplicated_countries =  np.delete(modified_countries, 0, axis=1)
duplicated_countries =  duplicated_countries[:,1:]


# Compute Correlations

correlation_matrix=np.zeros((countries_data.shape[1],countries_data.shape[1]),dtype=object)

for i in range(1,countries_data.shape[1]):
    for j in range(1,countries_data.shape[1]):
        corr=np.corrcoef(modified_countries[:, i].astype(float), modified_countries[:, j].astype(float))
        correlation_matrix[i,j]=corr[0,1]

corr_header=header[1:]
correlation_matrix[:,0]=header
correlation_matrix[0,:]=header
ctn.revert(correlation_matrix,'correlation.csv')



max_corr={}
# to calculate the max after because we already now corr[i,i]=1
np.fill_diagonal(correlation_matrix,0) 
for i in range(1,correlation_matrix.shape[1]):
    column=correlation_matrix[1:, i].astype(float)
    index_max=np.argmax(np.abs(column))
    value=column[index_max]
    category=correlation_matrix[index_max+1,0]
    max_corr[correlation_matrix[0,i]]={category:value}


ordered_corr={}
np.fill_diagonal(correlation_matrix,'avoid') 
for i in range(1,correlation_matrix.shape[1]):
    column_data={}
    column=correlation_matrix[1:, i]
    names=correlation_matrix[1:, 0]
    avoid_index=np.where(column == 'avoid')
    filtered_column=np.delete(column,avoid_index)
    filtered_names=np.delete(names,avoid_index)
    print(len(filtered_column))
    sort_indexes = np.argsort(np.absolute(filtered_column),axis=-1, kind="quicksort")
    for index in sort_indexes:
        column_data[filtered_names[index]]=filtered_column[index]
    ordered_corr[correlation_matrix[0,i]]=column_data




    

# Convert the list to a JSON string
json_str = json.dumps(max_corr)

# Open a file for writing
with open("list_file.json", "w") as f:
    # Write the JSON string to the file
    f.write(json_str)