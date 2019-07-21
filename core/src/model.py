from sklearn.base import BaseEstimator
import pysparnn.cluster_index as ci
from sklearn.metrics import pairwise_distances
import numpy as np



def get_close_users(matrix, new_user, top_n=3, metric='euclidean'):
    dist = pairwise_distances(matrix, new_user.reshape(1,-1), metric=metric)    
    dist = dist.flatten()
    idx = np.argsort(dist)[:top_n]    
    return matrix[idx]
	
def get_ids(new_user, closest, top_n=2):
    answer = []
    missing = np.where(new_user == 0)
    missing = np.array([x[1] for x in missing])
    columns_max = closest.max(axis=0)
    idx = np.argsort(columns_max)
    unsorted = np.intersect1d(missing, idx)
    buf = np.in1d(idx, missing)[::-1]
    for i,item in enumerate(buf):
        if item == True:
            answer.append(idx[::-1][i])
    
    return answer[:top_n]

# from annoy import AnnoyIndex
# class OrderBasedModel(BaseEstimator):
#   """https://github.com/spotify/annoy
#   """
#
#   def __init__(
#     self, n_neighbours=10, dims=100,
#     metric='angular', n_trees=10,
#     search_k=10
#   ):
#     self.n_neighbours = n_neighbours
#     self.n_trees = n_trees
#     self.search_k = search_k
#     self.annoy_index = AnnoyIndex(dims, metric)
#
#   def fit(self, X, y):
#     self.labels = y
#     for i, _ in enumerate(X):
#       self.annoy_index.add_item(i, X[i])
#     self.annoy_index.build(self.n_trees)
#     return self
#
#   def predict(self, X):
#     preds = self.annoy_index.get_nns_by_vector(
#       X, self.n_neighbours, self.search_k, True
#     )
#     return preds
#
#   def dump(self, filename):
#     self.annoy_index.save(filename)
#
#   def load(self, filename):
#     self.annoy_index.load(filename)

class OrderBasedModel(BaseEstimator):
  """https://github.com/facebookresearch/pysparnn
  """

  def __init__(self, n_neighbours=10):
    self.n_neighbours = n_neighbours

  def fit(self, X, y):
    self.cp = ci.MultiClusterIndex(X, y)

  def predict(self, X):
    self.cp.search(X, self.n_neighbours, True)


class UserBasedModel:
  pass


class DistancesModel:
  def __init__(self, n_closest=4, n_items, metric='euclidian'):
    self.top_n = top_n
    self.metric = metric

  def fit(self, X):
    self.matrix = X
	
	
	
  def get_ids(database, user, n_closest, n_items, metric):
	closest = get_close_users(database, user, n_closest, metric)
	indexes = predict(user, closest, n_items)
	return indexes

	

  def get_close_user(self, new_user):
      dist = pairwise_distances(self.matrix, new_user, metric=self.metric)        
      dist = dist.flatten()      
      idx = np.argsort(dist)[:self.top_n]          
      return self.matrix[idx]

  def predict(self, new_user):
    closest = self.get_close_user(new_user)
    missing = np.where(new_user == 0)[0]

    columns_max = closest.max(axis=0)
    idx = np.argsort(columns_max)
    return np.intersect1d(missing,idx)

  