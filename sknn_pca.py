# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:25:56 2017

@author: centraltendency
"""

 ## Sknn with PCA

import numpy as np
import pandas as pd
import logging
logging.basicConfig()

macro = pd.read_csv("/home/centraltendency/Kaggle/Sberbank/macro.csv")
train = pd.read_csv("/home/centraltendency/Kaggle/Sberbank/train.csv")
test = pd.read_csv("/home/centraltendency/Kaggle/Sberbank/train.csv")

# convert timestamp to time

train['timestamp'] = pd.to_datetime(train['timestamp'])

dataset = train.merge(macro, on = 'timestamp', how = 'left')
dataset.describe()
X = dataset.drop(["timestamp", "price_doc"], axis = 1)
y = dataset["price_doc"]

## One hot encoding data
from sklearn.preprocessing import LabelEncoder

for i in X.columns:
    if isinstance(X[i][0], str):
        print i
        le = LabelEncoder()
        X[i] = le.fit_transform(X[i])
        
## Impute missing data
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
imputed_X = pd.DataFrame(imp.fit_transform(X))
imputed_X.columns = X.columns
imputed_X.index = X.index

from sklearn.decomposition import PCA
pca = PCA()

pca.fit(imputed_X)
pca.explained_variance_ratio_

# Most of the data is explained by the first two principal components.

pca = PCA(n_components = 2).fit(imputed_X)
reduced_data = pca.transform(imputed_X)
reduced_data = pd.DataFrame(reduced_data, columns = ["Dimension 1", "Dimension 2"])

# Scale data

from sklearn.preprocessing import MinMaxScaler

for i in reduced_data.columns:
    reduced_data[i] = MinMaxScaler(feature_range = (0, 1.0)).fit_transform(reduced_data[i])
    
# SpliT data
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(reduced_data, y)
data = X_train.as_matrix()[:, np.newaxis]
target = y_train.as_matrix()[:, np.newaxis]  
data_test = X_test.as_matrix()[:, np.newaxis]
target_test = y_test.as_matrix()[:, np.newaxis]  

# train regressor

from sknn.mlp import Regressor, Layer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('neural network', Regressor(layers = [Layer("Sigmoid", units = 100), 
                                           Layer("Linear")], 
    learning_rate = 0.01, 
    n_iter = 10))    
    ])

# Gridserch

from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(pipeline, param_grid={
    'neural network__learning_rate': [0.00001, 0.0001, 0.001],
    'neural network__hidden0__units': [4, 8, 12, 14],
    'neural network__hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]})
gs.fit(data, target)
score = gs.score(data_test, target_test)
'{0:.10f}'.format(score)