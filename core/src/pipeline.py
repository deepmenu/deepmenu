
import pandas as pd

import numpy as np

from nltk.tokenize import RegexpTokenizer
from scipy.sparse import load_npz
from model import DistancesModel
import pickle
	
MENU_ITEMS = r'/home/misterion/Development/Datathon 2019/deepmenu/core/data/raw/MenuItem.csv'
EMBEDINGS = r'/home/misterion/Development/Datathon 2019/deepmenu/core/data/processed/embeddings_2201.npz'
LABELS = '/home/misterion/Development/Datathon 2019/deepmenu/core/data/processed/labels.npz'


VECT_VOCAB_PATH = '/home/misterion/Development/Datathon 2019/deepmenu/core/data/vectorizerVocab.pickle'
CAT_ITEM_PATH = '/home/misterion/Development/Datathon 2019/deepmenu/core/data/dictCatItemId.pickle'
ITEM_CAT_PATH = '/home/misterion/Development/Datathon 2019/deepmenu/core/data/dictItemIdCat.pickle'

class Pipeline:

    def __init__(self):
        self.tokenizer = RegexpTokenizer('[A-Za-z]+')
        self.df = pd.read_csv(MENU_ITEMS)
        self.df = self.df[['ID','ItemName', 'Description']]
    
        self.df = self.df[self.df['ID'].notnull()]
        self.df = self.df[self.df['ID'].apply(lambda x: x.isnumeric())]

        self.df['ID'] = self.df['ID'].astype(int)

        print(self.df.head())

        print(self.df.dtypes)

        X = load_npz(EMBEDINGS)
    
        self.model = DistancesModel(VECT_VOCAB_PATH, CAT_ITEM_PATH)
        self.model.fit(X)
	
    def id_to_category(self, id):
        row = self.df[self.df['ID'] == id]
        descr = (row['ItemName'] + row['Description']).iloc[0]

        with open(VECT_VOCAB_PATH, 'rb') as handle:
            voc = pickle.load(handle)

        return self.description2vector(descr, voc)

    def description2vector(self, description, cat2index):
        tokens = self.tokenizer.tokenize(description.lower())
        vct = np.zeros((len(cat2index),))
        for i, t in enumerate(tokens):
            if t not in cat2index:
                continue
            vct[cat2index[t]] += 1
        return vct

    def predict(self, id):
        vector = self.id_to_category(id)
        ids, cat = self.model.predict(vector.reshape(1,-1))
        
        rows = self.df[self.df['ID'].isin(ids)] 
        vals = rows.T.to_dict().values()
        return vals
