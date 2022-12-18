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
#from sklearn_extra.cluster import KMedoids

#réduction de dimensionnalité
from sklearn.decomposition import KernelPCA #ce n'est pas PCoA mais on peut l'utiliser pour que le résultat soit le même
from sklearn.manifold import Isomap


##-------------------------------------------ADULT----------------------------------------------------------#

##                                  DATASET DOWNLOAD AND CLEANUP


class DataParser:
  def __init__(self,database: str):
    data_features_db = pd.read_csv(database).to_numpy()
    self.data_features = np.delete(data_features_db,0,0)

  def splitData(self):
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

    return (training_x, test_x)
#                   MNIST


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


def adult_dissimilarity(x, y):
  numeric_feats_x = x[0,4,10,11,12]
  numeric_feats_y = y[0,4,10,11,12]
  #distance euclidienne sur features numériques
  num_dissimilarity = np.linalg.norm(numeric_feats_x - numeric_feats_y)
  avg = np.mean(num_dissimilarity)
  std = np.std(num_dissimilarity)
  num_dissimilarity /= std
  #estimation du poids pour les features catégoriques
  

  #usamos distancia euclidiana normalizada por valores numericos 
  #y le sumamos la fraccion de mismas valores categoricas * promedio 

  categoric_feats_x = x[1,5,6,7,8,9,13,14]
  categoric_feats_y = y[1,5,6,7,8,9,13,14]

  cat_dissimilarity = 0
  for i in range (len(categoric_feats_x)):
    if categoric_feats_x == categoric_feats_y:
      cat_dissimilarity += avg
  return num_dissimilarity + cat_dissimilarity
    
def numeric_distance_avg(X,Y):
  numeric_feats_x = X[0,4,10,11,12]
  numeric_feats_y = Y[0,4,10,11,12]
  dissimilarity = np.linalg.norm(numeric_feats_x - numeric_feats_y)
  return np.mean(dissimilarity)

def numeric_distance_std(X,Y):
  numeric_feats_x = X[0,4,10,11,12]
  numeric_feats_y = Y[0,4,10,11,12]
  dissimilarity = np.linalg.norm(numeric_feats_x - numeric_feats_y)
  return np.std(dissimilarity, axis=0)


def adult_dissimilarity_matrix(X,Y):
  return adult_dissimilarity(X, Y)


#dp = DataParser()
  


  



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
dp = DataParser(database='D1/mnist.csv')
mnist_train,mnist_test=dp.splitData()

trainSimilarity = similarity_matrix(mnist_train)
testSimilarity = similarity_matrix(mnist_test)

plt.figure(figsize=(6,6)).add_subplot(111).imshow(similarity)

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