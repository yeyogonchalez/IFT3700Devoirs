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
    data_features_db = pd.read_csv('D1/adult.csv').to_numpy()
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
  numeric_feats_x = np.array(x[:, [0,4,10,11,12]])
  numeric_feats_y = np.array(y[:, [0,4,10,11,12]]) 
  categoric_feats_x = x[:, [1,5,6,7,8,9,13,14]]
  categoric_feats_y = y[:, [1,5,6,7,8,9,13,14]]

  #standardisation des données, soustraction de la moyenne innécessaire
  #car il s'agit d'une différence
  delta = np.subtract(numeric_feats_x, numeric_feats_y)
  delta = np.divide(delta, std)
  #distance euclidienne sur features numériques
  num_dissimilarity = np.linalg.norm(delta, axis=1)
  
  #estimation du poids pour les features catégoriques
  global_num_avg = np.mean(avg)
  
  #calcul de la dissimilarité 
  n_common_feats =  np.sum(categoric_feats_x == categoric_feats_y, axis=1)
  cat_dissimilarity = global_num_avg * n_common_feats
  return num_dissimilarity + cat_dissimilarity

def adult_dissimilarity_matrix(X,Y):
    dis_matrix = np.zeros(shape=(len(X),len(Y)))
    for i in range()


def numeric_distance_avg(X):
  numeric_feats_x = X[:, [0,4,10,11,12]]
  return np.mean(numeric_feats_x)

def numeric_distance_std(X,Y):
  numeric_feats_x = X[:, [0,4,10,11,12]]
  return np.std(numeric_feats_x, axis=0)


def adult_dissimilarity_matrix(X,Y):
  return adult_dissimilarity(X, Y)


dp = DataParser()
train_set, test_set = dp.splitData()
#adult_dissimilarity(train_set[0], train_set[1] )
adult_dissimilarity(train_set[:19536], train_set[19536:39072])



  



##----------------------------------------------------------------------------------------------------------#



##-------------------------------------------MNIST----------------------------------------------------------#

