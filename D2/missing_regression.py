import numpy as np
import csv_to_np as ctn # support script that converts csv to np array
from sklearn.linear_model import ElasticNet

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
    
duplicated_countries = modified_countries.copy().reshape(countries_data.shape[0], countries_data.shape[1]*2)
#duplicated_countries[:0] =   countries_data[:,0]
for i in range(1,countries_data.shape[1]):
    # assign the values of the original column to the first half of the new column
    duplicated_countries[:, i*2] = modified_countries[:, i]
    median = statistics[1,i]
    
    for j in range(countries_data.shape[0]):
        if duplicated_countries[j, i*2]<=median:
            duplicated_countries[j,i*2+1]=0
        else:
            duplicated_countries[j,i*2+1]=1
    

    # assign the values of the original column to the second half of the new column
    

ctn.revert(duplicated_countries,'duplicated_countries.csv')