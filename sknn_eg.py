# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:47:20 2017

@author: centraltendency
"""

## Neural Nets

import numpy as np
import pandas as pd
import logging
logging.basicConfig()

from pandas.tools.plotting import scatter_matrix
from pandas.tools.plotting import andrews_curves
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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

## standardize data
from sklearn.preprocessing import MinMaxScaler

for i in X.columns:
    imputed_X[i] = MinMaxScaler(feature_range = (0, 1.0)).fit_transform(imputed_X[i])

# PCA
# apply pca by fitting the data with the same number of dimensions as features

from sklearn.decomposition import PCA
#pca = PCA()
#
#pca.fit(imputed_X)
#pca.explained_variance_ratio_

## Most of the data is explained by the first two principal components.

#pca = PCA(n_components = 7).fit(imputed_X)
#reduced_data = pca.transform(imputed_X)
#reduced_data = pd.DataFrame(reduced_data, columns = ["Dimension 1", "Dimension 2", "Dimension 3", "Dimension 4", "Dimension 5", "Dimension 6", "Dimension 7"])
#
### Split data
#from sklearn.model_selection import train_test_split
#
#X_train, X_test, y_train, y_test = train_test_split(reduced_data, y)
#
### Random Forest
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import r2_score
#from sklearn.metrics import mean_squared_error
#
#clf_rf = RandomForestRegressor()
#clf_rf.fit(X_train, y_train)
#
#clf_r2 = r2_score(y_test, clf_rf.predict(X_test)) # 0.34062819632035346
#mse = mean_squared_error(y_test, clf_rf.predict(X_test)) # 14539149684678.707
#clf_r2
#mse
#
## Neural Networks
from sknn.mlp import Regressor, Layer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('neural network', Regressor(layers = [Layer("Sigmoid", units = 100), 
                                           Layer("Linear")], 
    learning_rate = 0.00001, 
    n_iter = 10))    
    ])
#    
#data = X_train.as_matrix()[:, np.newaxis]
#target = y_train.as_matrix()[:, np.newaxis]   
#pipeline.fit(data, target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(imputed_X, y)
data = X_train.as_matrix()[:, np.newaxis]
target = y_train.as_matrix()[:, np.newaxis]  
 
pipeline.fit(data, target)
score = pipeline.score(X_test.as_matrix()[:, np.newaxis], y_test.as_matrix()[:, np.newaxis])

#pipeline.score(X_test.as_matrix()[:, np.newaxis], y_test.as_matrix())
#r2_score(y_test.as_matrix(), pipeline.predict(X_test.as_matrix()[:, np.newaxis]))

# Grid Search
from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(pipeline, param_grid={
    'neural network__hidden0__units': [4, 8, 12, 14],
    'neural network__hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]})
gs.fit(data, target)