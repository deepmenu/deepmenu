
import pandas as pd

from scipy.sparse import load_npz
from model import DistancesModel


MENU_ITEMS = '/home/misterion/Development/Datathon 2019/deepmenu/core/data/processed/menu_final.txt'
EMBEDINGS = '/home/misterion/Development/Datathon 2019/deepmenu/core/data/processed/embedings.npz'
LABELS = '/home/misterion/Development/Datathon 2019/deepmenu/core/data/processed/labels.npz'


class Pipeline:
    def __init__(self):
        self.df = pd.read_csv(MENU_ITEMS)
        self.df = self.df[['ID','ItemName']]
        self.df.columns= ['MenuItemID','Description']
        self.df = self.df[self.df['MenuItemID'].apply(lambda x: x.isnumeric())]
        self.df['MenuItemID'] = self.df['MenuItemID'].astype(int)      

        X = load_npz(EMBEDINGS)
        
        # y = pd.read_csv(LABELS_FILEPATH)['user_id'].values

        self.model = DistancesModel()
        self.model.fit(X)

    def id_to_descrition(self, id):
        return self.df[self.df['MenuItemID'] == id]

    def description_to_vector(self):
        #VECTORIZER
        pass

    def predict_category(self, vector):
        self.model.predict(vector)

    def get_most_popular_items(self, category):
        pass
