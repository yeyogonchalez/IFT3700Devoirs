import numpy as np
import csv_to_np as ctn # support script that converts csv to np array
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import json


def data_parser(x, y):
    # Get the size of the input data
    size = len(x)

    # Initialize empty lists for the training and test data
    training_features = []
    training_labels = []
    test_features = []
    test_labels = []

    # Iterate over the data and split it into training and test sets
    for i in range(size):
        if i % 7 == 1:
            # Append the feature and label to the test lists
            test_features.append(x[i])
            test_labels.append(y[i])
        else:
            # Append the feature and label to the training lists
            training_features.append(x[i])
            training_labels.append(y[i])

    # Convert the training and test lists to NumPy arrays
    train_x = np.array(training_features)
    train_y = np.array(training_labels)
    test_x = np.array(test_features)
    test_y = np.array(test_labels)
    
    # Return the training and test data as NumPy arrays
    return train_x, train_y, test_x, test_y


countries_data=ctn.convert('devoir2_code/filtered_countries_data.csv')

statistics=ctn.convert('devoir2_code/statistics.csv')

# number of regressions we are gonna use for the linear regression
num_regressions=2

# Get the header row and remove it from the list of rows
header = countries_data[0]
countries_data = countries_data[1:,:]
statistics=statistics[1:,:]


# used as a separate to variable to store the predictions separately
modified_countries=countries_data.copy() 
#---------------------------------------------------------------Q1.d-------------------------------------------------------
# Stage 0
for i in range(1,countries_data.shape[1]):
  column = countries_data[:, i]
  median = statistics[1,i]

  modified_countries[:,i] = np.where(column=='NA', median, column)


for i in range(num_regressions):
    # Iterate over the columns in the data
    for j in range(1, countries_data.shape[1]):
        # Initialize an empty list for the predictions
        predictions = []

        # Get the data for the current column
        y = modified_countries[:, j]

        # Convert the data to float
        y = y.astype(float)

        # Get the data for all other columns
        X = np.delete(modified_countries, j, axis=1)

        # Remove the first column of data (the countries)
        X = X[:, 1:]

        # Convert the data to float
        X = X.astype(float)

        # Get the indices of the rows with missing data in the current column
        indices = np.where(countries_data[:, j] == 'NA')[0]

        # Remove the rows with missing data from the training data
        y_train = np.delete(y, indices, axis=0)
        X_train = np.delete(X, indices, axis=0)

        # Get the rows with missing data for the current column
        X_pred = X[indices]
        lr = ElasticNet()
        lr.fit(X, y)

        # Iterate over the rows with missing data
        for index in indices:
            # Make a prediction for the current row
            prediction = lr.predict(X[index].reshape(1, -1))
            
            # Update the modified data with the prediction
            modified_countries[index][j] = prediction[0]

# when not in commentary use to get a csv with modified_countries data 
#ctn.revert(modified_countries,'output.csv') 
#---------------------------------------------------------------Q1.e-------------------------------------------------------
# Duplicate columns
duplicated_countries = np.empty((countries_data.shape[0], countries_data.shape[1]*2),dtype=object)
for i in range(1, countries_data.shape[1]):
    # Assign the values of the original column to the first half of the new column
    duplicated_countries[:, i*2] = modified_countries[:, i]
    # Get the median value of the original column
    median = float(statistics[1, i])
    # Assign the values of the original column to the second half of the new column
    for j in range(countries_data.shape[0]):
        if float(duplicated_countries[j, i*2]) <= median:
            # If the value is less than or equal to the median, assign 0
            duplicated_countries[j, i*2+1] = 0
        else:
            # If the value is greater than the median, assign 1
            duplicated_countries[j, i*2+1] = 1
    

    
    
duplicated_countries[:,1] =  modified_countries[:,0]

#duplicated_countries =  np.delete(modified_countries, 0, axis=1)
#duplicated_countries =  duplicated_countries[:,1:]

# when not in commentary use to get a csv with duplicated_countries 
#ctn.revert(duplicated_countries,'output.csv') 


#---------------------------------------------------------------Q2.a-------------------------------------------------------
# Compute Correlations
correlation_matrix=np.zeros((countries_data.shape[1],countries_data.shape[1]),dtype=object)

for i in range(1,countries_data.shape[1]):
    for j in range(1,countries_data.shape[1]):
        corr=np.corrcoef(modified_countries[:, i].astype(float), modified_countries[:, j].astype(float))
        correlation_matrix[i,j]=corr[0,1]

corr_header=header[1:]
correlation_matrix[:,0]=header
correlation_matrix[0,:]=header

# when not in commentary use to get a csv with correlation_matrix
#ctn.revert(correlation_matrix,'correlation.csv')


#---------------------------------------------------------------Q2.b-------------------------------------------------------
max_corr={}
# to calculate the max after because we already now corr[i,i]=1
np.fill_diagonal(correlation_matrix,0) 

for i in range(1, correlation_matrix.shape[1]):
    # Get the data for the current column
    column = correlation_matrix[1:, i].astype(float)
    # Find the index of the maximum absolute value in the column
    index_max = np.argmax(np.abs(column))
    # Get the maximum absolute value
    value = column[index_max]
    # Get the category name at the row with the maximum value
    category = correlation_matrix[index_max+1, 0]
    # Add the category and value to the dictionary
    max_corr[correlation_matrix[0, i]] = {category: value}


#---------------------------------------------------------------Q2.c-------------------------------------------------------
ordered_corr={}
np.fill_diagonal(correlation_matrix,'avoid') 
for i in range(1, correlation_matrix.shape[1]):
    # Initialize an empty dictionary for the current column
    column_data = {}
    # Get the data for the current column
    column = correlation_matrix[1:, i]
    # Get the names of the categories
    names = correlation_matrix[1:, 0]
    # Get the indices of the rows with the value 'avoid'
    avoid_index = np.where(column == 'avoid')[0]
    # Remove the rows with the value 'avoid'
    filtered_column = np.delete(column, avoid_index)
    filtered_names = np.delete(names, avoid_index)
    # Sort the data by the absolute value of the correlation
    sort_indexes = np.argsort(np.absolute(filtered_column), axis=-1, kind="quicksort")
    # Iterate over the sorted indices
    for index in sort_indexes:
        # Add the category name and correlation value to the dictionary
        column_data[filtered_names[index]] = filtered_column[index]
    # Add the dictionary for the current column to the overall dictionary
    ordered_corr[correlation_matrix[0, i]] = column_data

#---------------------------------------------------------------Q3.c-------------------------------------------------------
precision={}
for j in range(1, countries_data.shape[1]):
    # Initialize an empty list to store the predictions
    predictions = []

    # Get the data for the current column
    y_lr = modified_countries[:, j]
    y_class = duplicated_countries[:, j*2+1]
    y_lr = y_lr.astype(float)
    y_class = y_class.astype(int)

    # Get the features
    X = np.delete(modified_countries, j, axis=1)
    X = X[:, 1:]
    X = X.astype(float)

    # Split the data into training and test sets
    train_x, train_y_lr, test_x, test_y_lr = data_parser(X, y_lr)
    train_y_class = data_parser(X, y_class)[1]
    test_y_class = data_parser(X, y_class)[3]

    # Fit and predict with a linear regression model
    lr = LinearRegression()
    lr.fit(train_x, train_y_lr)
    lr_predictions = lr.predict(test_x)

    # Calculate the mean absolute error
    mae = mean_absolute_error(test_y_lr, lr_predictions)

    # Fit and predict with a Gaussian Naive Bayes model
    nb = GaussianNB()
    nb.fit(train_x, train_y_class)
    nb_predictions = nb.predict(test_x)

    # Calculate the accuracy
    accuracy = metrics.accuracy_score(test_y_class, nb_predictions)
    

    precision[correlation_matrix[0,j]]={'linear regression MAE':mae,"Bayes Classifier accuracy":accuracy}

'''
# Convert the dictionary to a JSON string
json_str = json.dumps(precision) 

# Open a file for writing
with open("list_file.json", "w") as f:
    # Write the JSON string to the file
    f.write(json_str)
'''


#---------------------------------------------------------------Q3.b-------------------------------------------------------

# Best pair of columns

# Initialize a dictionary to store the accuracies for each pair of predictor columns
accuracies = {}
accuracies_avg={}
accuracy_list=[]
# Iterate over all possible pairs of predictor columns
for i in range(1, countries_data.shape[1]):
    # Initialize the minimum mean absolute error and the maximum accuracy
    mae_min = [None, None, np.inf]
    accuracy_max = [None, None, 0]

    # Iterate over all pairs of predictor columns
    for j in range(1, countries_data.shape[1]-1):
        if i == j:
            continue
        for k in range(j+1, countries_data.shape[1]):
            if i == k:
                continue

            # Get the indices of the predictor columns
            predictor_columns = [j, k]
            permuted_x = modified_countries[1:, predictor_columns]
            permuted_x = permuted_x.astype(float)

            # Get the data for the current column
            y_lr = modified_countries[:, i]
            y_class = duplicated_countries[:, i*2+1] 
            y_lr = y_lr.astype(float)
            y_class = y_class.astype(int)

            # Split the data into training and test sets
            train_x, train_y_lr, test_x, test_y_lr = data_parser(permuted_x, y_lr)
            train_y_class = data_parser(permuted_x, y_class)[1]
            test_y_class = data_parser(permuted_x, y_class)[3]

            # Fit and predict with a linear regression model
            lr = LinearRegression()
            lr.fit(train_x, train_y_lr)
            lr_predictions = lr.predict(test_x)

            # Calculate the mean absolute error
            mae = mean_absolute_error(test_y_lr, lr_predictions)

            # Fit and predict with a Naive Bayes classifier
            nb = GaussianNB()
            nb.fit(train_x, train_y_class)
            nb_predictions = nb.predict(test_x)
            accuracy = metrics.accuracy_score(test_y_class, nb_predictions)

            # Convert the predictor columns to a tuple
            predictor_columns_tuple = tuple(predictor_columns)

            # Add the accuracy to the dictionary using the indices of the predictor columns as the key
            if predictor_columns_tuple in accuracies_avg:
                accuracies_avg[predictor_columns_tuple].append(mae)
            else:
                accuracies_avg[predictor_columns_tuple] = [mae]

            # Update the minimum mean absolute error
            if mae < mae_min[2]:
                mae_min = [j, k, mae]

            # Update the maximum accuracy
            if accuracy > accuracy_max[2]:
                accuracy_max = [j, k, accuracy]
    # Create a dictionary with the results for the current column
    data = {
        'MAE column 1': header[mae_min[0]],
        "MAE column 2": header[mae_min[1]],
        'MAE': mae_min[2],
        'Accuracy column 1': header[accuracy_max[0]],
        "Accuracy column 2": header[accuracy_max[1]],
        'Accuracy': accuracy_max[2]
    }

    # Add the results to the dictionary
    accuracies[header[i]] = data

    # Add the indices of the predictor columns with the best accuracy and minimum MAE to the list
    accuracy_list.append([mae_min[0], mae_min[1]])
    accuracy_list.append([accuracy_max[0], accuracy_max[1]])


#---------------------------------------------------------------Q3.c-------------------------------------------------------

average_accuracies = {k: sum(v) / len(v) for k, v in accuracies_avg.items()}

# Find the key with the maximum average accuracy
max_key = min(average_accuracies, key=lambda k: average_accuracies[k])

# Get the maximum average accuracy
max_accuracy = average_accuracies[max_key]

#print(max_accuracy,header[max_key[0]],header[max_key[1]])




# Initialize a counter for each list
counters = {tuple(lst): 0 for lst in accuracy_list}

# Count the number of times each list appears in the list of lists
for lst in accuracy_list:
    counters[tuple(lst)] += 1

# Find the list that appears the most times
most_common_list = max(counters, key=counters.get)

#print(most_common_list,counters[most_common_list])

most_common_list=[header[most_common_list[0]],header[most_common_list[1]]]
#print(most_common_list)
    

# Convert the dictionary to a JSON string
json_str = json.dumps(accuracies) 

# Open a file for writing
with open("accuracies.json", "w") as f:
    # Write the JSON string to the file
    f.write(json_str)
