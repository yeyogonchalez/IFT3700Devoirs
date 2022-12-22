import numpy as np
import csv_to_np as ctn # support script that converts csv to np array
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import json


def data_parser(x,y):
    size = len(x)
    training_features = []
    training_labels=[]
    test_features = []
    test_labels=[]

    for i in range(size):
        if i%7==1:
          test_features.append(x[i])
          test_labels.append(y[i])
        else:
          training_features.append(x[i])
          training_labels.append(y[i])

    train_x = np.array(training_features)
    train_y = np.array(training_labels)
    test_x = np.array(test_features)
    test_y = np.array(test_labels)
    
    return train_x,train_y,test_x,test_y 

countries_data=ctn.convert('D2/filtered_countries_data.csv')

statistics=ctn.convert('D2/statistics.csv')

num_regressions=2

# Get the header row and remove it from the list of rows
header = countries_data[0]
countries_data = countries_data[1:,:]
statistics=statistics[1:,:]


# used as a separate to variable to store the predictions separately
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
#duplicated_countries =  duplicated_countries[:,1:]


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
    sort_indexes = np.argsort(np.absolute(filtered_column),axis=-1, kind="quicksort")
    for index in sort_indexes:
        column_data[filtered_names[index]]=filtered_column[index]
    ordered_corr[correlation_matrix[0,i]]=column_data

precision={}
for j in range(1,countries_data.shape[1]):
    predictions=[]
    y_lr = modified_countries[:, j]
    y_class = duplicated_countries[:,j*2+1] 
    y_lr=y_lr.astype(float)
    y_class=y_class.astype(int)

    X = np.delete(modified_countries, j, axis=1)

    X=X[:,1:]

    X=X.astype(float)
    
    train_x,train_y_lr,test_x,test_y_lr = data_parser(X,y_lr)

    train_y_class = data_parser(X,y_class)[1]

    test_y_class = data_parser(X,y_class)[3]

    lr=LinearRegression()
    lr.fit(train_x,train_y_lr)
    lr_predictions=lr.predict(test_x)

    mae=mean_absolute_error(test_y_lr,lr_predictions)

    nb=GaussianNB()
    nb.fit(train_x,train_y_class)
    nb_predictions=nb.predict(test_x)
    accuracy=metrics.accuracy_score(test_y_class,nb_predictions)
    

    precision[correlation_matrix[0,j]]={'linear regression MAE':mae,"Bayes Classifier accuracy":accuracy}


# Convert the dictionary to a JSON string
json_str = json.dumps(precision) 

# Open a file for writing
with open("list_file.json", "w") as f:
    # Write the JSON string to the file
    f.write(json_str)



# Best pair of columns

# Initialize a dictionary to store the accuracies for each pair of predictor columns
accuracies = {}
accuracy_list=[]
# Iterate over all possible pairs of predictor columns
for i in range(1,countries_data.shape[1]):
    mae_min=[None,None,np.inf]
    accuracy_max=[None,None,0]
    for j in range(1, countries_data.shape[1]-1):
        if i==j:
            continue
        for k in range(j+1,countries_data.shape[1]):
            if i==k:
                continue
            # Get the indices of the predictor columns
            predictor_columns = [j, k]
            permuted_x=modified_countries[1:,[j,k]]
            permuted_x=permuted_x.astype(float)

            y_lr = modified_countries[:, i]
            y_class = duplicated_countries[:,i*2+1] 

            y_lr=y_lr.astype(float)
            y_class=y_class.astype(int)

            train_x,train_y_lr,test_x,test_y_lr = data_parser(permuted_x,y_lr)

            train_y_class = data_parser(permuted_x,y_class)[1]

            test_y_class = data_parser(permuted_x,y_class)[3]

            lr=LinearRegression()
            lr.fit(train_x,train_y_lr)
            lr_predictions=lr.predict(test_x)

            mae=mean_absolute_error(test_y_lr,lr_predictions)
            

            nb=GaussianNB()
            nb.fit(train_x,train_y_class)
            nb_predictions=nb.predict(test_x)
            accuracy=metrics.accuracy_score(test_y_class,nb_predictions)


            if mae<mae_min[2]:
                mae_min=[j,k,mae]

            if (accuracy>accuracy_max[2]):
                accuracy_max=[j,k,accuracy]       
    data={'MAE column 1':header[mae_min[0]],"MAE column 2":header[mae_min[1]],'MAE':mae_min[2],'Accuracy column 1':header[accuracy_max[0]],"Accuracy column 2":header[accuracy_max[1]],'Accuracy':accuracy_max[2]}        
    accuracies[header[i]]=data    
    accuracy_list.append([mae_min[0],mae_min[1]])
    accuracy_list.append([accuracy_max[0],accuracy_max[1]])


# Initialize a counter for each list
counters = {tuple(lst): 0 for lst in accuracy_list}

# Count the number of times each list appears in the list of lists
for lst in accuracy_list:
    counters[tuple(lst)] += 1

# Find the list that appears the most times
most_common_list = max(counters, key=counters.get)

#print(most_common_list,counters[most_common_list])

most_common_list=[header[most_common_list[0]],header[most_common_list[1]]]
print(most_common_list)
    

# Convert the dictionary to a JSON string
json_str = json.dumps(accuracies) 

# Open a file for writing
with open("accuracies.json", "w") as f:
    # Write the JSON string to the file
    f.write(json_str)
