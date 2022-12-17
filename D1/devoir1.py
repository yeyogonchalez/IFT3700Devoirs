##                                           IMPORTS                                                       
import numpy as np
import matplotlib as plt
import scipy as sp 
import pandas as pd

#classification
from sklearn.neighbors import KNeighborsClassifier #k-plus proches voisins

#Partitionnement
from sklearn.cluster import AgglomerativeClustering #Regroupement hiérarchique (Paritionnement binaire)
#from pyclustering.cluster.kmedoids import kmedoids

#réduction de dimensionnalité
from sklearn.decomposition import KernelPCA #ce n'est pas PCoA mais on peut l'utiliser pour que le résultat soit le même
from sklearn.manifold import Isomap


##-------------------------------------------ADULT----------------------------------------------------------#

##                                  DATASET DOWNLOAD AND CLEANUP


class DataParser:
  def __init__(self):
    data_features_db = pd.read_csv('adult.csv').to_numpy()
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


dp = DataParser()
  


  



##----------------------------------------------------------------------------------------------------------#



##-------------------------------------------MNIST----------------------------------------------------------#

