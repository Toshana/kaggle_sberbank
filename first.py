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
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

macro = pd.read_csv("/home/centraltendency/Kaggle/Sberbank/macro.csv")
train = pd.read_csv("/home/centraltendency/Kaggle/Sberbank/train.csv")
test = pd.read_csv("/home/centraltendency/Kaggle/Sberbank/train.csv")

# convert timestamp to time
train['timestamp'] = pd.to_datetime(train['timestamp'])

dataset = train.merge(macro, on = 'timestamp', how = 'left')
dataset.describe


fig, ax = plt.subplots()
ax.plot(train.timestamp, train.price_doc)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

scatter_matrix(train)

train.plot.box()
