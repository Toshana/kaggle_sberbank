# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:54:16 2017

@author: centraltendency
"""
#import sys
#sys.path.remove('/usr/lib/python2.7/dist-packages')

## Predict realty prices (price_doc)

import numpy as np
import pandas as pd
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
X = dataset.drop(["id", "timestamp"], axis = 1)
y = dataset["id"]

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
from sklearn.preprocessing import StandardScaler

for i in X.columns:
    imputed_X[i] = StandardScaler().fit_transform(imputed_X[i])

#for i in dataset.columns:
#        plt.figure()
#        plt.plot(dataset['timestamp'], dataset[i])
#        plt.xlabel(i)
#        plt.ylabel("Time")
#        name = i + ".png"    
#        plt.savefig(name)
#        plt.close()
#
#fig, ax = plt.subplots()
#ax.plot(train.timestamp, train.price_doc)
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#
#scatter_matrix(train)
#
#for i in train.columns:
#    if type(i) != int:
#        pass
#    else:
#        train[i].astype(float)
#
#plt.figure()
#andrews_curves(train, 'timestamp')


# PCA
# apply pca by fitting the data with the same number of dimensions as features

from sklearn.decomposition import PCA
pca = PCA()

pca.fit(imputed_X)
pca.explained_variance_
