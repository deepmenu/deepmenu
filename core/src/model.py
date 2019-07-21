from sklearn.base import BaseEstimator
import pysparnn.cluster_index as ci


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
