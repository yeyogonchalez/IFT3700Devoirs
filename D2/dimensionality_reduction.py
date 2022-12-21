from sklearn.manifold import Isomap
import csv_to_np as ctn # support script that converts csv to np array
import matplotlib.pyplot as plt
import numpy as np

##---------------------------------------------A------------------------------------------------------
#loading data
countries_data = ctn.convert('D2/regression_files/missing_regression_2_lr.csv')
countries_num_data = countries_data[:,1:]
countries_names = countries_data[:,0]

#find best hyperparameter
n_neighbors_list = range(5,150,4)
reconstruction_errors2 = []
reconstruction_errors5 = []
for n_neighbors in n_neighbors_list:
    isomap2 = Isomap(n_neighbors=n_neighbors, n_components=2)
    isomap2.fit_transform(countries_num_data)
    reconstruction_errors2.append(isomap2.reconstruction_error())
    isomap5 = Isomap(n_neighbors=n_neighbors, n_components=5)
    isomap5.fit_transform(countries_num_data)
    reconstruction_errors5.append(isomap5.reconstruction_error())

plt.plot(n_neighbors_list, reconstruction_errors2, color='red', label="2 components")
plt.plot(n_neighbors_list, reconstruction_errors5, color='blue', label="5 components")
plt.xlabel("number of neighbors")
plt.ylabel("reconstruction error")
plt.legend(loc='upper center')
plt.show()
plt.clf()

min_constr_error_index = np.argmin(reconstruction_errors2)
star_n_neighbors = n_neighbors_list[min_constr_error_index]

isomap = Isomap(n_neighbors=n_neighbors, n_components=2)
reduced_feats = np.asarray(isomap.fit_transform(countries_num_data))
reduced_feats_x = list(reduced_feats[:,0])
reduced_feats_y = list(reduced_feats[:,1])

#visualize dataset in 2D
fig, ax = plt.subplots()
fig.set_size_inches(28,18)
plt.scatter(reduced_feats_x, reduced_feats_y)
plt.xlabel("x")
plt.ylabel("y")
for i, country_name in enumerate(countries_names):
    a= reduced_feats_x[0]
    ax.annotate(country_name, (reduced_feats_x[i], reduced_feats_y[i]) )
plt.show()





