# -*- coding: utf-8 -*-
"""
Created on Sat May 13 20:04:42 2017

@author: centraltendency
"""

## Load Data

import numpy as np
import pandas as pd

macro = pd.read_csv("/home/centraltendency/Kaggle/Sberbank/macro.csv")
train = pd.read_csv("/home/centraltendency/Kaggle/Sberbank/train.csv")
test = pd.read_csv("/home/centraltendency/Kaggle/Sberbank/train.csv")
dataset = train.merge(macro, on = 'timestamp', how = 'left')

time = train['timestamp']
price = train['price_doc']
ts = pd.concat([time, price], axis = 1)
ts = pd.DataFrame(ts)
ts["timestamp"] = pd.to_datetime(ts["timestamp"])

import matplotlib.pyplot as plt
plt.plot(ts['timestamp'], ts['price_doc'])

##  modern_education_share: Share of state (municipal) educational organizations, 
## corresponding to modern requirements of education in the total number of high schools;
## old_education_build_share: The share of state (municipal) educational organizations, 
## buildings are in disrepair and in need of major repairs of the total number.
## Most of these are nan. Drop from dataframe.

dropped = ["modern_education_share", "old_education_build_share", "timestamp", "id", "price_doc"]
dataset.drop(dataset[dropped], axis = 1, inplace = True)

data_str_list = []
data_num_list = []
for i in dataset.columns:
    if isinstance(dataset[i][0], str):
        data_str_list.append(i)
    else:
        data_num_list.append(i)

data_num = dataset[data_num_list]
data_str = dataset[data_str_list]

## Pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
str_data = data_str.apply(encoder.fit_transform)
full = pd.concat([str_data, data_num], axis = 1)

num_pipeline = Pipeline([
    ('imputer', Imputer(strategy = 'mean')),
    ('minmaxscaler', MinMaxScaler(feature_range = (0, 1.0)))
])

data = num_pipeline.fit_transform(full)

