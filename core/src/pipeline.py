
import pandas as pd
import numpy as np
import pickle

from nltk.tokenize import RegexpTokenizer
from scipy.sparse import load_npz
from core.src.model import DistancesModel
from core.src.entity import MenuItem
	
MENU_ITEMS = r'/home/misterion/Development/Datathon 2019/deepmenu/core/data/raw/MenuItem.csv'
EMBEDINGS = r'/home/misterion/Development/Datathon 2019/deepmenu/core/data/processed/embeddings_2201.npz'


VECT_VOCAB_PATH = r'/home/misterion/Development/Datathon 2019/deepmenu/core/data/vectorizerVocab.pickle'
CAT_ITEM_PATH = r'/home/misterion/Development/Datathon 2019/deepmenu/core/data/dictCatItemId.pickle'
ITEM_CAT_PATH = r'/home/misterion/Development/Datathon 2019/deepmenu/core/data/dictItemIdCat.pickle'

class Pipeline:

    def __init__(self):
        self.tokenizer = RegexpTokenizer('[A-Za-z]+')
        self.df = pd.read_csv(MENU_ITEMS)
        self.current_index = 0

        self.__clean_df()

        X = load_npz(EMBEDINGS)
    
        self.model = DistancesModel(VECT_VOCAB_PATH, CAT_ITEM_PATH)
        self.model.fit(X)


    def get_items(self, start='next', end=20):
        if start == 'next':
            start = self.current_index
        self.current_index = start + end
        vals = self.df.iloc[start:start + end].T.to_dict().values()
        items = []
        for value in vals:
            items.append(MenuItem(value['ID'],value['ItemName'], value['Description']))
        return items


    def predict(self, id):
        vector = self.__id_to_category(id)
        ids, cat = self.model.predict(vector.reshape(1,-1))
        
        rows = self.df[self.df['ID'].isin(ids)] 
        vals = rows.T.to_dict().values()
        items = []
        for value in vals:
            items.append(MenuItem(value['ID'],value['ItemName'], value['Description'], cat))
        return items


    def __clean_df(self):
        self.df = self.df[['ID','ItemName', 'Description']]         
        self.df = self.df.dropna()
        self.df = self.df[self.df['ID'].apply(lambda x: x.isnumeric())]
        self.df['ID'] = self.df['ID'].astype(int)


    def __id_to_category(self, id):
        row = self.df[self.df['ID'] == id]
        descr = (row['ItemName'] + row['Description']).iloc[0]

        with open(VECT_VOCAB_PATH, 'rb') as handle:
            voc = pickle.load(handle)

        return self.__description2vector(descr, voc)

    def __description2vector(self, description, cat2index):
        tokens = self.tokenizer.tokenize(description.lower())
        vct = np.zeros((len(cat2index),))
        for i, t in enumerate(tokens):
            if t not in cat2index:
                continue
            vct[cat2index[t]] += 1
        return vct

pipe = Pipeline()
pipe.predict(433)
pipe.predict(455)