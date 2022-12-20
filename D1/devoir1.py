##                                           IMPORTS                                                       
import numpy as np
import matplotlib.pyplot as plt
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

import time

##-------------------------------------------SHARED----------------------------------------------------------#

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

##                                  DIMENSIONALITY REDUCTION EVALUATION
def visualize_2D(reduced_feats_train, reduced_feats_test, train_colors, test_colors):
  reduced_feats_train_x = reduced_feats_train[:, 0]
  reduced_feats_train_y = reduced_feats_train[:, 1]
  reduced_feats_test_x = reduced_feats_test[:, 0]
  reduced_feats_test_y = reduced_feats_test[:, 1]
  fig = plt.figure(figsize=(12, 8))
  ax = fig.add_subplot(211)
  ax.set_title('Training')
  ax.scatter(reduced_feats_train_x, reduced_feats_train_y, c=train_colors)
  ax = fig.add_subplot(212)
  ax.set_title('Testing')
  ax.scatter(reduced_feats_test_x, reduced_feats_test_y, c=test_colors)
  plt.show()

def isomap(train_set_dissimilarity, test_set_dissimilarity, n_components, train_colors, test_colors):
  isomap = Isomap(n_components = n_components, metric='precomputed')
  isomap_train = isomap.fit_transform(train_set_dissimilarity)
  isomap_test = isomap.transform(test_set_dissimilarity)
  train_colors = np.where(train_labels == '<=50K', "red", "green")
  test_colors = np.where(test_labels == '<=50K', "red", "green")
  visualize_2D(isomap_train, isomap_test, train_colors, test_colors)

def pcoa(train_set_dissimilarity, test_set_dissimilarity, n_components, train_colors, test_colors):
  pcoa = KernelPCA(n_components, kernel='precomputed')
  pcoa_train = pcoa.fit_transform(-.5*train_set_dissimilarity**2)
  pcoa_test = pcoa.transform(-.5*test_set_dissimilarity**2) 
  visualize_2D(pcoa_train, pcoa_test, train_colors, test_colors)

##                                     CLUSTERING EVALUATION
def compute_silhouette_score(dissimilarity_matrix, clusters):
    scores = []
    for i in range(len(dissimilarity_matrix)):
        cluster = find_cluster(i, clusters)
        a = np.mean([dissimilarity_matrix[i][j] for j in cluster if j!=i])
        inter2 = [np.mean([dissimilarity_matrix[i][j] for j in clusters[k]]) for k in range(len(clusters)) if k != i]
        b = np.min(inter2)
        score = (b - a) / max(a, b)
        scores.append(score)
    return np.mean(scores)

def find_cluster(i, clusters):
  for cluster in clusters:
    if i in cluster:
      return cluster

def clusterize(list, n_clusters):
  clusters = []
  for i in range(n_clusters):
    clusters.append([])
  for j in range(len(list)):
    clusters[list[j]].append(j)
  return clusters
    
def agglomerative_clustering_predict(agglomerative_clustering, dissimilarity_matrix):
    average_dissimilarity = list()
    for i in range(agglomerative_clustering.n_clusters):
        ith_clusters_dissimilarity = dissimilarity_matrix[:, np.where(agglomerative_clustering.labels_==i)[0]]
        average_dissimilarity.append(ith_clusters_dissimilarity.mean(axis=1))
    return np.argmin(np.stack(average_dissimilarity), axis=0)

def k_medoids(train_set_dissimilarity, test_set_dissimilarity, k):
  medoids = range(k)
  kmedoids_instance = kmedoids(train_set_dissimilarity, medoids, data_type='distance_matrix')
  kmedoids_instance.process() #training
  kmedoid_train_clusters = kmedoids_instance.predict(train_set_dissimilarity)
  kmedoid_train_clusters = clusterize(kmedoid_train_clusters, len(medoids))
  kmedoid_test_clusters = kmedoids_instance.predict(test_set_dissimilarity)
  kmedoid_test_clusters = clusterize(kmedoid_test_clusters, len(medoids))

  #EVALUATION
  kmedoid_train_silhouette_scores = silhouette(train_set_dissimilarity, kmedoid_train_clusters).process().get_score()
  kmedoid_test_silhouette_scores = silhouette(test_set_dissimilarity, kmedoid_test_clusters).process().get_score()
  kmedoid_train_silhouette_score = np.mean(kmedoid_train_silhouette_scores)
  kmedoid_test_silhouette_score = np.mean(kmedoid_test_silhouette_scores)
  print("le score silhouette de kmedoides pour le jeu d'entraînement est: ", kmedoid_train_silhouette_score)
  print("le score silhouette de kmedoides pour le jeu de test est: ", kmedoid_test_silhouette_score)

def agglomerative_clustering(train_set_dissimilarity, test_set_dissimilarity, n_clusters):
  agglomerative_clustering = AgglomerativeClustering(n_clusters, affinity='precomputed', linkage='average')
  agglomerative_clustering.fit(train_set_dissimilarity)
  agglo_train_clusters = agglomerative_clustering_predict(agglomerative_clustering, train_set_dissimilarity)
  agglo_train_clusters = clusterize(agglo_train_clusters, n_clusters)
  agglo_test_clusters = agglomerative_clustering_predict(agglomerative_clustering, test_set_dissimilarity)
  agglo_test_clusters = clusterize(agglo_test_clusters, n_clusters)

  agglo_train_silhouette_scores = silhouette(train_set_dissimilarity, agglo_train_clusters).process().get_score()
  agglo_test_silhouette_scores = silhouette(test_set_dissimilarity, agglo_test_clusters).process().get_score()
  agglo_train_silhouette_score = np.mean(agglo_train_silhouette_scores)
  agglo_test_silhouette_score = np.mean(agglo_test_silhouette_scores)
  print("le score silhouette de agglo clustering pour le jeu d'entraînement est: ", agglo_train_silhouette_score)
  print("le score silhouette de agglo clustering pour le jeu de test est: ", agglo_test_silhouette_score)


##                                  CLASSIFICATION EVALUATION

def compute_error_rate(class_preds, true_classes):
    error_num = np.sum(class_preds != true_classes)
    return error_num / len(true_classes)

def knn(train_set_dissimilarity, test_set_dissimilarity, n_neighbors, train_labels, test_labels):
  knn = KNeighborsClassifier(n_neighbors, metric='precomputed', algorithm='brute')
  knn.fit(train_set_dissimilarity, train_labels)
  knn_train_predictions = knn.predict(train_set_dissimilarity)
  train_error = compute_error_rate(train_labels, knn_train_predictions)
  knn_test_predictions = knn.predict(test_set_dissimilarity)
  test_error = compute_error_rate(test_labels, knn_test_predictions)
  print("le taux de précision pour le jeu d'entraînement est: ", 1-train_error)
  print("le taux de précision pour le jeu de test est: ", 1-test_error)

##----------------------------------------------------------------------------------------------------------#
##-------------------------------------------ADULT----------------------------------------------------------#



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
train_set, test_set = dp.splitData(100)
start = time.time()
train_set_dissimilarity = adult_dissimilarity_matrix(train_set, train_set, dp.get_features(), recompute=True)
end = time.time()
print(end-start , " seconds elapsed")
test_set_dissimilarity = adult_dissimilarity_matrix(test_set, train_set, dp.get_features(), recompute=True)
n_components = 2
k=2
n_clusters = 2
n_neighbors = 2
train_labels = train_set[:, 14]
test_labels = test_set[:, 14]
train_colors = np.where(train_labels == '<=50K', "red", "green")
test_colors = np.where(test_labels == '<=50K', "red", "green")

#                                          ISOMAP
isomap(train_set_dissimilarity, test_set_dissimilarity, n_components, train_colors, test_colors)

##                                          PCOA
pcoa(train_set_dissimilarity, test_set_dissimilarity, n_components, train_colors, test_colors)

##                                          K-MEDOIDS
k_medoids(train_set_dissimilarity, test_set_dissimilarity, k)

##                                           REGROUPEMENT HIÉRARCHIQUE 
agglomerative_clustering(train_set_dissimilarity, test_set_dissimilarity, n_clusters)


##                                           KNN 
knn(train_set_dissimilarity, test_set_dissimilarity, n_neighbors, train_labels, test_labels)

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
  return alpha * euclidean_distance + (1 - alpha) * cosine_dist

##                                          SIMILARITY MATRIX
#  points: size nxm Our points to compare 
#
#  return a matrix of n x n containing the combined similarity between each pair of points
def dissimilarity_matrix(points):
  n = points.shape[0]
  diss_matrix = np.empty((n, n))
  for i in range(n):
    for j in range(n):
      diss_matrix[i][j] = similarity(points[i], points[j])

  return diss_matrix

##                                          INIT
dp = DataParser(database='D1/mnist.csv')
mnist_train, mnist_test = dp.splitData(1000)
mnist_train_dissimilarity = dissimilarity_matrix(mnist_train)
mnist_test_dissimilarity = dissimilarity_matrix(mnist_test)
mnist_n_components = 10
mnist_k=10
mnist_n_clusters = 10
mnist_n_neighbors = 5
mnist_train_labels = mnist_train[:, 0]
mnist_test_labels = mnist_test[:, 0]
colors = np.array(["red","green","blue","yellow","pink","orange","purple","brown","cyan","magenta"])
mnist_train_colors = [colors[i] for i in mnist_train_labels]
mnist_test_colors = [colors[i] for i in mnist_test_labels]

##                                          ISOMAP
isomap(mnist_train_dissimilarity, mnist_test_dissimilarity, mnist_n_components, mnist_train_colors, mnist_test_colors)

##                                          PCOA
pcoa(mnist_train_dissimilarity, mnist_test_dissimilarity, mnist_n_components, mnist_train_colors, mnist_test_colors)

##                                          K-MEDOIDS
k_medoids(mnist_train_dissimilarity, mnist_test_dissimilarity, mnist_k)

##                                          AGGLOMERATIVE CLUSTERING
agglomerative_clustering(mnist_train_dissimilarity, mnist_test_dissimilarity, mnist_n_clusters)

##                                          KNN
knn(mnist_train_dissimilarity, mnist_test_dissimilarity, mnist_n_neighbors, mnist_train_labels, mnist_test_labels)