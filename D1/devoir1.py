##                                           IMPORTS                                                       
import numpy as np
import matplotlib as plt
import scipy as sp 
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine

#classification
from sklearn.neighbors import KNeighborsClassifier #k-plus proches voisins

#Partitionnement
from sklearn.cluster import AgglomerativeClustering #Regroupement hiérarchique (Paritionnement binaire)
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.silhouette import silhouette
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids

#réduction de dimensionnalité
from sklearn.decomposition import KernelPCA #ce n'est pas PCoA mais on peut l'utiliser pour que le résultat soit le même
from sklearn.manifold import Isomap


##-------------------------------------------ADULT----------------------------------------------------------#

##                                  DATASET DOWNLOAD AND CLEANUP


class DataParser:
  def __init__(self,database: str):
    data_features_db = pd.read_csv(database).to_numpy()
    self.data_features = np.delete(data_features_db,0,0)

  def get_features(self):
    return self.data_features

  def splitData(self, sample_size):
    features = self.data_features
    size = len(features)
    training_features = []
    test_features = []

    for i in range(size):
      if i%5==1:
        test_features.append(features[i])
      else:
        training_features.append(features[i])

    training_x = np.array(training_features)
    test_x = np.array(test_features)

    #échantillonage
    training_x = training_x[np.random.choice(training_x.shape[0], np.floor(sample_size*4/5).astype(int), replace=False), :]
    test_x = test_x[np.random.choice(test_x.shape[0], np.floor(sample_size*0.2).astype(int), replace=False), :]

    return (training_x, test_x)



##                                          SIMILARITY

# 0, age: num
# 1, workclass: cat
# 2, fnlwgt: num QUE PEDO CON ESTA
# 3, education: cat
# 4, education-num: num
# 5, marital-status: cat
# 6, occupation: cat
# 7, relationship: cat
# 8, race: cat
# 9, sex: cat
# 10, capital-gain: num
# 11, capital-loss: num
# 12, hours-per-week: num
# 13, native-country: cat
# 14, class, cat


def adult_dissimilarity(x, y, avg, std):

  #initialisation
  numeric_feats_x = np.array([x[index] for index in (0,4,10,11,12)]).astype(float)
  numeric_feats_y = np.array([y[index] for index in (0,4,10,11,12)]).astype(float)
  categoric_feats_x = np.array([x[index] for index in (1,5,6,7,8,9,13,14)])
  categoric_feats_y = np.array([y[index] for index in (1,5,6,7,8,9,13,14)])

  #standardisation des données, soustraction de la moyenne innécessaire
  #car il s'agit d'une différence de points
  delta = np.subtract(numeric_feats_x, numeric_feats_y)
  delta = np.divide(delta, std)
  #distance euclidienne sur features numériques
  num_dissimilarity = np.linalg.norm(delta)
  
  #estimation du poids pour les features catégoriques
  avg_std_num_dissimilarity = np.linalg.norm(avg/std) 
  weight = avg_std_num_dissimilarity / len(numeric_feats_x)
  
  #calcul de la dissimilarité 
  n_different_feats =  np.sum(categoric_feats_x != categoric_feats_y, axis=0)
  cat_dissimilarity = weight * n_different_feats
  return num_dissimilarity + cat_dissimilarity

def adult_dissimilarity_matrix(X, Y, D, recompute):
  #on assume que le test_set sera toujours plus petit que le train_set
  if len(X) == len(Y):
    file_name = 'train_diss_matrix.npy'
  else:
    file_name = 'test_diss_matrix.npy'
  
  if recompute:
    diss_matrix = np.zeros(shape=(len(X),len(Y)))
    avg = numeric_distance_avg(D)
    std = numeric_distance_std(D)
    for i in range(len(X)):
      for j in range(len(Y)):
        diss_matrix[i,j] = adult_dissimilarity(X[i], Y[j], avg, std)
    np.save(file_name, diss_matrix)
  else:
    diss_matrix = np.load(file_name)
  return diss_matrix


def numeric_distance_avg(X):
  numeric_feats_x = X[:, [0,4,10,11,12]].astype(float)
  return np.mean(numeric_feats_x, axis=0)

def numeric_distance_std(X):
  numeric_feats_x = X[:, [0,4,10,11,12]].astype(float)
  return np.std(numeric_feats_x, axis=0)







##                                          INIT
dp = DataParser("D1/adult.csv")
train_set, test_set = dp.splitData(1000)
train_set_dissimilarity = adult_dissimilarity_matrix(train_set, train_set, dp.get_features(), recompute=True)
test_set_dissimilarity = adult_dissimilarity_matrix(test_set, train_set, dp.get_features(), recompute=True)


##                                          ISOMAP, MISSING PROPER ANALYSIS
# isomap = Isomap(n_components=1, metric='precomputed')
# isomap_train = isomap.fit_transform(train_set_dissimilarity)
# isomap_test = isomap.transform(test_set_dissimilarity)

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(211)
# ax.set_title('Train Isomap')
# ax.scatter(isomap_train, np.zeros_like(isomap_train))

# ax = fig.add_subplot(212)
# ax.set_title('Cosine Isomap sur infinity')
# ax.scatter(isomap_test, np.zeros_like(isomap_test))




# ##                                          PCOA
# pcoa = KernelPCA(n_components=1, kernel='precomputed')
# pcoa_circle = pcoa.fit_transform(-.5*train_set_dissimilarity**2)
# pcoa_infinity = pcoa.transform(-.5*test_set_dissimilarity**2) 


##                                          K-MEDOIDS
initial_medoids = [0,1,2,3]
kmedoids_instance = kmedoids(train_set_dissimilarity, initial_medoids, data_type='distance_matrix')
kmedoids_instance.process() #training
clusters = kmedoids_instance.get_clusters()

kmedoids_train = kmedoids_instance.predict(train_set_dissimilarity)
kmedoids_test = kmedoids_instance.predict(test_set_dissimilarity)



#silhouette score
# silhouette_train_score = silhouette(train_set_dissimilarity, clusters, data_type='distance_matrix').process().get_score()
# silhouette_test_score = silhouette(test_set_dissimilarity, clusters, data_type='distance_matrix').process().get_score()
def compute_silhouette_score(dissimilarity_matrix, clusters):
    # Calculate the silhouette score for each sample
    scores = []
    for i, cluster in enumerate(clusters):
        # Calculate the average dissimilarity of the sample to other samples in its own cluster
        a = np.mean([dissimilarity_matrix[i][j] for j in cluster])
        
        # Calculate the average dissimilarity of the sample to samples in the next nearest cluster
        b = np.min([np.mean([dissimilarity_matrix[i][j] for j in clusters[k]]) for k in range(len(clusters)) if k != i])
        
        # Calculate the silhouette score for the sample
        score = (b - a) / max(a, b)
        scores.append(score)
    
    # Return the average silhouette score for the dataset
    return np.mean(scores)

train_silhouette_score = compute_silhouette_score(train_set_dissimilarity, clusters)
test_silhouette_score = compute_silhouette_score(test_set_dissimilarity, clusters)
print(silhouette_score)


# ##                                           REGROUPEMENT HIÉRARCHIQUE 
# def agglomerative_clustering_predict(agglomerative_clustering, dissimilarity_matrix):
#     average_dissimilarity = list()
#     for i in range(agglomerative_clustering.n_clusters):
#         ith_clusters_dissimilarity = dissimilarity_matrix[:, np.where(agglomerative_clustering.labels_==i)[0]]
#         average_dissimilarity.append(ith_clusters_dissimilarity.mean(axis=1))
#     return np.argmin(np.stack(average_dissimilarity), axis=0)

# agglomerative_clustering = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage='average')
# agglomerative_clustering.fit(train_set_dissimilarity)

# agglo_circle = agglomerative_clustering_predict(agglomerative_clustering, train_set_dissimilarity)
# agglo_infinity = agglomerative_clustering_predict(agglomerative_clustering, test_set_dissimilarity)



# quadrants = 1
# knn = KNeighborsClassifier(n_neighbors=2, metric='precomputed', algorithm='brute')
# knn.fit(train_set_dissimilarity, quadrants)

# knn_circle = knn.predict(train_set_dissimilarity)
# knn_infinity = knn.predict(test_set_dissimilarity)

##----------------------------------------------------------------------------------------------------------#



##-------------------------------------------MNIST----------------------------------------------------------#


##                                          SIMILARITY
#  X: size m.     Vector data point 
#  Y: size n.     Vector data point
#  alpha: coefficient of weight of the euclidian distance
#
#  Returns the gradient matrix of W which is of size m*n

def similarity(X, Y,alpha: float=0.5):

  euclidean_distance = euclidean(X, Y)

  cosine_dist = cosine(X, Y)
  cosine_sim = 1 - cosine_dist

  return alpha * euclidean_distance + (1 - alpha) * cosine_sim

##                                          SIMILARITY MATRIX
#  points: size nxm Our points to compare 
#
#  return a matrix of n x n containing the combined similarity between each pair of points
def similarity_matrix(points):
  n = points.shape[0]
  sim_matrix = np.empty((n, n))
  for i in range(n):
    for j in range(n):
      sim_matrix[i][j] = similarity(points[i], points[j])
    print('punto '+str(i)+'/'+str(n))
  return sim_matrix

##                                          TEST
# dp = DataParser(database='D1/mnist.csv')
# mnist_train,mnist_test=dp.splitData()

# trainSimilarity = similarity_matrix(mnist_train)
# testSimilarity = similarity_matrix(mnist_test)

# plt.figure(figsize=(6,6)).add_subplot(111).imshow(similarity)

##                                          K-NN
# Create an instance of the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, metric='precomputed', algorithm='brute')

# Fit the model to the training data
knn.fit(trainSimilarity)

# Use the model to make predictions on the test data
predictions = knn.predict(mnist_test)

##                                          K-Medoids
initial_medoids = [0,1,2] # no se que pedo con esto
# Create an instance of the K-Medoids model
kmedoids_instance = kmedoids(trainSimilarity, initial_medoids, data_type='distance_matrix')

# Fit the model to the data
kmedoids_instance.process() #training


# Predict the cluster labels for the data
kmedoids_instance.predict(mnist_test)

##                                          PCoA
pcoa = KernelPCA(n_components=1, kernel='precomputed')
pcoa_circle = pcoa.fit_transform(-.5*trainSimilarity**2) #-.5*D**2 est crucial!!!
pcoa_infinity = pcoa.transform(-.5*testSimilarity**2) #-.5*D**2 est crucial!!!

##                                          Isomap
isomap = Isomap(n_components=1, n_neighbors=2, metric='precomputed')
isomap_circle = isomap.fit_transform(trainSimilarity)
isomap_infinity = isomap.transform(testSimilarity)