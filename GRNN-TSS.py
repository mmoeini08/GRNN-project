# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 19:47:15 2022

@author: mmoein2
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from neupy.algorithms import RBFKMeans
from neupy.algorithms import GRNN
import seaborn as sns
sns.set(style='darkgrid')
file_name = 'TSS'
data_raw = pd.read_csv(file_name + '.csv', header=0, index_col=0)
print("{0} rows and {1} columns".format(len(data_raw.index), len(data_raw.columns)))
print("")
data = data_raw#.drop(['Name'], axis=1)
data_train, data_test = train_test_split(data, test_size=0.2)
minval = data_train.min()
minmax = data_train.max() - data_train.min()
data_train_scaled = (data_train - minval) / minmax
data_test_scaled = (data_test - minval) / minmax
X_train = data_train_scaled.drop(['TSS'], axis=1)
Y_train = data_train_scaled.TSS
X_test = data_test_scaled.drop(['TSS'], axis=1)
Y_test = data_test_scaled.TSS
GRNNet = GRNN(std=0.1, verbose=False) #Learn more at http://neupy.com/apidocs/neupy.algorithms.rbfn.grnn.html
GRNNet.train(X_train, Y_train)
score = cross_val_score(GRNNet, X_train, Y_train, scoring='r2', cv=5)
print("")
print("Cross Validation: {0} (+/- {1})".format(score.mean().round(2), (score.std() * 2).round(2)))
print("")
Y_predict = GRNNet.predict(X_test)
print(Y_test.values * minmax.TSS + minval.TSS)
print("")
print((Y_predict * minmax.TSS + minval.TSS)[:,0].round(2))
print("")
print("Accuracy: {0}".format(metrics.r2_score(Y_test, Y_predict).round(2)))
print("")