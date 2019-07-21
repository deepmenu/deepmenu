import pandas as pd
import numpy as np
from os.path import join
from scipy.sparse import load_npz

from core.src.model import OrderBasedModel


DATA_FILEPATH = '/home/maaxap/workspace/data/menuby/embedings.npz'
LABELS_FILEPATH = '/home/maaxap/workspace/data/menuby/labels.csv'


def main():
  X = load_npz(DATA_FILEPATH)
  y = pd.read_csv(LABELS_FILEPATH)['user_id'].values

  x_train, y_train = X[:-100], y[:-100]
  x_test, y_test = X[-100:], y[-100:]

  model = OrderBasedModel()
  model.fit(x_train, y)

  preds = model.predict(X[0])
  print(preds)


if __name__ == '__main__':
  main()