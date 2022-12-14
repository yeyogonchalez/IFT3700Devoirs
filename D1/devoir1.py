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
  def __init__(self,database: str, recompute, data_id):
    data_features_db = pd.read_csv(database).to_numpy()
    self.data_features = np.delete(data_features_db,0,0)
    self.recompute = recompute
    if data_id == "ADULT":
      self.train_file_name = 'D1/npyFiles/adult_train_set.npy'
      self.test_file_name = 'D1/npyFiles/adult_test_set.npy'
    else:
      self.train_file_name = 'D1/npyFiles/mnist_train_set.npy'
      self.test_file_name = 'D1/npyFiles/mnist_test_set.npy'

  def get_features(self):
    return self.data_features

  #lors de la division en jeu d'entraînement et de test, on échantillonne pour des fins pratiques
  def splitData(self, sample_size):
    if self.recompute:
      features = self.data_features
      size = len(features)
      training_features = []
      test_features = []

      for i in range(size):
        if i%5==1:
          test_features.append(features[i])
        else:
          training_features.append(features[i])

      train_set = np.array(training_features)
      test_set = np.array(test_features)

      #échantillonage
      train_set = train_set[np.random.choice(train_set.shape[0], np.floor(sample_size*4/5).astype(int), replace=False), :]
      test_set = test_set[np.random.choice(test_set.shape[0], np.floor(sample_size*0.2).astype(int), replace=False), :]
      np.save(self.train_file_name, train_set)
      np.save(self.test_file_name, test_set)
    else:
      train_set = np.load(self.train_file_name, allow_pickle=True)
      test_set = np.load(self.test_file_name, allow_pickle=True)

    return (train_set, test_set)

##                                  DIMENSIONALITY REDUCTION EVALUATION
def visualize_2D(reduced_feats_train, reduced_feats_test, train_colors, test_colors, file_name):
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
  plt.savefig(file_name)
  plt.clf()

def isomap(train_set_dissimilarity, test_set_dissimilarity, n_components, train_colors, test_colors, file_name):
  isomap = Isomap(n_components = n_components, metric='precomputed')
  isomap_train = isomap.fit_transform(train_set_dissimilarity)
  isomap_test = isomap.transform(test_set_dissimilarity)
  visualize_2D(isomap_train, isomap_test, train_colors, test_colors, file_name)
  
def pcoa(train_set_dissimilarity, test_set_dissimilarity, n_components, train_colors, test_colors, file_name):
  pcoa = KernelPCA(n_components, kernel='precomputed')
  pcoa_train = pcoa.fit_transform(-.5*train_set_dissimilarity**2)
  pcoa_test = pcoa.transform(-.5*test_set_dissimilarity**2) 
  visualize_2D(pcoa_train, pcoa_test, train_colors, test_colors, file_name)


##                                     CLUSTERING EVALUATION
#finds to which cluster the point belongs to
def find_cluster(i, clusters):
  for cluster in clusters:
    if i in cluster:
      return cluster

#transforms a list into a list of clusters
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

def k_medoids(train_set_dissimilarity, test_set_dissimilarity, k, file_name):
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
  kmedoid_train_silhouette_score = np.nanmean(kmedoid_train_silhouette_scores)
  kmedoid_test_silhouette_score = np.nanmean(kmedoid_test_silhouette_scores)

  if file_name != "":
    file = open(file_name, "w+")
    file.write("le score silhouette de kmedoides pour le jeu d'entrainement est: " + str(kmedoid_train_silhouette_score) + "\n")
    file.write("le score silhouette de kmedoides pour le jeu de test est: " + str(kmedoid_test_silhouette_score) + "\n")
    file.close() 
  return kmedoid_train_silhouette_score, kmedoid_test_silhouette_score

def agglomerative_clustering(train_set_dissimilarity, test_set_dissimilarity, n_clusters, file_name):
  agglomerative_clustering = AgglomerativeClustering(n_clusters, affinity='precomputed', linkage='average')
  agglomerative_clustering.fit(train_set_dissimilarity)
  agglo_train_clusters = agglomerative_clustering_predict(agglomerative_clustering, train_set_dissimilarity)
  agglo_train_clusters = clusterize(agglo_train_clusters, n_clusters)
  agglo_test_clusters = agglomerative_clustering_predict(agglomerative_clustering, test_set_dissimilarity)
  agglo_test_clusters = clusterize(agglo_test_clusters, n_clusters)

  agglo_train_silhouette_scores = silhouette(train_set_dissimilarity, agglo_train_clusters).process().get_score()
  agglo_test_silhouette_scores = silhouette(test_set_dissimilarity, agglo_test_clusters).process().get_score()
  agglo_train_silhouette_score = np.nanmean(agglo_train_silhouette_scores)
  agglo_test_silhouette_score = np.nanmean(agglo_test_silhouette_scores)
  if file_name != "":
    file = open(file_name, "w+")
    file.write("le score silhouette de agglo clustering pour le jeu d'entrainement est: "+ str(agglo_train_silhouette_score) + "\n")
    file.write("le score silhouette de agglo clustering pour le jeu de test est: " + str(agglo_test_silhouette_score) + "\n")
    file.close()

  return agglo_train_silhouette_score, agglo_test_silhouette_score


##                                  CLASSIFICATION EVALUATION

def compute_error_rate(class_preds, true_classes):
    error_num = np.sum(class_preds != true_classes)
    return error_num / len(true_classes)

def knn(train_set_dissimilarity, test_set_dissimilarity, n_neighbors, train_labels, test_labels, file_name):
  knn = KNeighborsClassifier(n_neighbors, metric='precomputed', algorithm='brute')
  knn.fit(train_set_dissimilarity, train_labels)
  knn_train_predictions = knn.predict(train_set_dissimilarity)
  train_error = compute_error_rate(train_labels, knn_train_predictions)
  knn_test_predictions = knn.predict(test_set_dissimilarity)
  test_error = compute_error_rate(test_labels, knn_test_predictions)
  if file_name != "":
    file = open(file_name, "w+")
    file.write("le taux de precision pour le jeu d'entrainement est: " + str(1-train_error) + "\n")
    file.write("le taux de precision pour le jeu de test est: " + str(1-test_error) + "\n")
    file.close()
  return 1-train_error, 1-test_error

##                                  PLOTTING TOOLS

def saveplot(x1, y1, x2, y2, xlabel, ylabel, file_name):
  plt.plot(x1, y1, color='green')
  plt.plot(x2,y2, color='red')
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.savefig(file_name)
  plt.clf()

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
    matrix_file_name = 'D1/npyFiles/adult_train_diss_matrix.npy'
  else:
    matrix_file_name = 'D1/npyFiles/adult_test_diss_matrix.npy'
  
  if recompute:
    diss_matrix = np.zeros(shape=(len(X),len(Y)))
    avg = numeric_distance_avg(D)
    std = numeric_distance_std(D)
    for i in range(len(X)):
      for j in range(len(Y)):
        diss_matrix[i,j] = adult_dissimilarity(X[i], Y[j], avg, std)
    np.save(matrix_file_name, diss_matrix)
  else:
    diss_matrix = np.load(matrix_file_name)
  return diss_matrix

def numeric_distance_avg(X):
  numeric_feats_x = X[:, [0,4,10,11,12]].astype(float)
  return np.mean(numeric_feats_x, axis=0)

def numeric_distance_std(X):
  numeric_feats_x = X[:, [0,4,10,11,12]].astype(float)
  return np.std(numeric_feats_x, axis=0)


##                                          INIT

##SUPER IMPORTANT PARAMETER:
adult_recompute = True
############################
dp = DataParser("D1/adult.csv", adult_recompute, data_id = "ADULT")
train_set, test_set = dp.splitData(100)
start = time.time()
train_set_dissimilarity = adult_dissimilarity_matrix(train_set, train_set, dp.get_features(), adult_recompute)
end = time.time()
print(end-start , " seconds elapsed")
test_set_dissimilarity = adult_dissimilarity_matrix(test_set, train_set, dp.get_features(), adult_recompute)
n_components = 2
k=2
n_clusters = 2
n_neighbors = 2
train_labels = train_set[:, 14]
test_labels = test_set[:, 14]
train_colors = np.where(train_labels == '<=50K', "red", "green")
test_colors = np.where(test_labels == '<=50K', "red", "green")

#                                          ISOMAP
isomap(train_set_dissimilarity, test_set_dissimilarity, n_components, train_colors, test_colors, "D1/AdultResults/isomap.png")


##                                          PCOA
pcoa(train_set_dissimilarity, test_set_dissimilarity, n_components, train_colors, test_colors, "D1/AdultResults/pcoa.png")

##                                          K-MEDOIDS
k_medoids(train_set_dissimilarity, test_set_dissimilarity, k, "D1/AdultResults/kmedoids.txt")

##                                           REGROUPEMENT HIÉRARCHIQUE 
agglomerative_clustering(train_set_dissimilarity, test_set_dissimilarity, n_clusters, "D1/AdultResults/agglo_clust.txt")


##                                           KNN 
knn(train_set_dissimilarity, test_set_dissimilarity, n_neighbors, train_labels, test_labels, "D1/AdultResults/knn.txt")

##----------------------------------------------------------------------------------------------------------#



##-------------------------------------------MNIST----------------------------------------------------------#


##                                          SIMILARITY
#  X: size m.     Vector data point 
#  Y: size n.     Vector data point
#  alpha: coefficient of weight of the euclidian distance
#
#  Returns the gradient matrix of W which is of size m*n

def mnist_dissimilarity(X, Y, alpha):
  euclidean_distance = euclidean(X, Y)
  cosine_dist = cosine(X, Y)
  return alpha * euclidean_distance + (1 - alpha) * cosine_dist

##                                          SIMILARITY MATRIX
#  points: size nxm Our points to compare 
#
#  return a matrix of n x n containing the combined similarity between each pair of points
def mnist_dissimilarity_matrix(X, Y, recompute, alpha):
  if len(X) == len(Y):
    matrix_file_name = 'D1/npyFiles/mnist_train_diss_matrix.npy'
  else:
    matrix_file_name = 'D1/npyFiles/mnist_test_diss_matrix.npy'
  
  if recompute:
    diss_matrix = np.zeros(shape=(len(X),len(Y)))
    for i in range(len(X)):
      for j in range(len(Y)):
        diss_matrix[i,j] = mnist_dissimilarity(X[i], Y[j], alpha)
    np.save(matrix_file_name, diss_matrix)
  else:
    diss_matrix = np.load(matrix_file_name)
  return diss_matrix

##                                          INIT

##SUPER IMPORTANT PARAMETER
mnist_recompute = True
###########################
dp = DataParser('D1/mnist.csv', mnist_recompute, data_id = "MNIST" )
mnist_train, mnist_test = dp.splitData(100)
alpha = 1
start = time.time()
mnist_train_dissimilarity = mnist_dissimilarity_matrix(mnist_train, mnist_train, mnist_recompute, alpha)
end = time.time()
print(end-start, " seconds for matrix computation")
mnist_test_dissimilarity = mnist_dissimilarity_matrix(mnist_test, mnist_train, mnist_recompute, alpha)
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
isomap(mnist_train_dissimilarity, mnist_test_dissimilarity, mnist_n_components, mnist_train_colors, mnist_test_colors, "D1/MnistResults/isomap.png" )

##                                          PCOA
pcoa(mnist_train_dissimilarity, mnist_test_dissimilarity, mnist_n_components, mnist_train_colors, mnist_test_colors, "D1/MnistResults/pcoa.png")

##                                          K-MEDOIDS
k_medoids(mnist_train_dissimilarity, mnist_test_dissimilarity, mnist_k, "D1/MnistResults/kmedoids.txt")

##                                          AGGLOMERATIVE CLUSTERING
agglomerative_clustering(mnist_train_dissimilarity, mnist_test_dissimilarity, mnist_n_clusters, "D1/MnistResults/agglo_clust.txt")

##                                          KNN
knn(mnist_train_dissimilarity, mnist_test_dissimilarity, mnist_n_neighbors, mnist_train_labels, mnist_test_labels, "D1/MnistResults/knn.txt")

##                                          COMPARISON WITH EUCLIDIAN DISTANCE
alphas = [0, 0.2, 0.4, 0.6, 0.8, 1]
kmedoids_train_silh_scores = []
agglo_train_silh_scores = []
knn_train_accuracies = []
kmedoids_test_silh_scores = []
agglo_test_silh_scores = []
knn_test_accuracies = []
i = 0
comparison_recompute = True
for alpha in alphas:
  mnist_train_dissimilarity = mnist_dissimilarity_matrix(mnist_train, mnist_train, comparison_recompute, alpha)
  mnist_test_dissimilarity = mnist_dissimilarity_matrix(mnist_test, mnist_train, comparison_recompute, alpha)

  ##                                          ISOMAP
  isomap(mnist_train_dissimilarity, mnist_test_dissimilarity, mnist_n_components, mnist_train_colors, mnist_test_colors, "D1/MnistComparison/isomap" + str(i) + ".png" )

  ##                                          PCOA
  pcoa(mnist_train_dissimilarity, mnist_test_dissimilarity, mnist_n_components, mnist_train_colors, mnist_test_colors, "D1/MnistComparison/pcoa" + str(i) + ".png")

  ##                                          K-MEDOIDS
  kmedoids_train_silh_score, kmedoids_test_silh_score = k_medoids(mnist_train_dissimilarity, mnist_test_dissimilarity, mnist_k, "")
  kmedoids_train_silh_scores.append(kmedoids_train_silh_score)
  kmedoids_test_silh_scores.append(kmedoids_test_silh_score)

  ##                                          AGGLOMERATIVE CLUSTERING
  agglo_train_silh_score, agglo_test_silh_score = agglomerative_clustering(mnist_train_dissimilarity, mnist_test_dissimilarity, mnist_n_clusters, "")
  agglo_train_silh_scores.append(agglo_train_silh_score)
  agglo_test_silh_scores.append(agglo_test_silh_score)

  ##                                           KNN
  knn_train_accuracy, knn_test_accuracy =  knn(mnist_train_dissimilarity, mnist_test_dissimilarity, mnist_n_neighbors, mnist_train_labels, mnist_test_labels, "")
  knn_train_accuracies.append(knn_train_accuracy)
  knn_test_accuracies.append(knn_test_accuracy)

  i+=1

saveplot(alphas, kmedoids_train_silh_scores, alphas, kmedoids_test_silh_scores, "alpha", "silhouette score", "D1/MnistComparison/kmedoids.png")
saveplot(alphas, agglo_train_silh_scores, alphas, agglo_test_silh_scores, "alpha", "silhouette score", "D1/MnistComparison/agglo_clust.png")
saveplot(alphas, knn_train_accuracies, alphas, knn_test_accuracies, "alpha", "accuracy rate", "D1/MnistComparison/knn.png")




