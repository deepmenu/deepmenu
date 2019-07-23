import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from collections import Counter


# some prototypes for item-based mining

def read(path):

	df = pd.read_csv(path)
	# do smth	
	return df
	

	
def get_traces(data):
	traces = df.groupby(groups).sum()[transaction]
	l = []
	for item in traces.values:
		l.append([i for i in item])
	
	return lambda
	

def encode(data):
	te = TransactionEncoder()
	te_ary = te.fit(l).transform(l)
	df = pd.DataFrame(te_ary, columns=te.columns_)
	return df
	
	
def apriori(df, support):
	frequent_itemsets = apriori(df, min_support = support, use_colnames=True)
	frequent_itemsets['abs_support'] = frequent_itemsets['support']*len(l)
	frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
	return frequent_itemsets