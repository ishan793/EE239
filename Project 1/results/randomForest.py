# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:03:42 2016

@author: shubham
"""

import cPickle as pickle
data = pickle.load(open("data/housing_data.pickle","rb"))
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
import numpy as np

#import matplotlib.pyplot as plt

mydata = np.asarray(data['x']).astype(np.float)
mylabel = np.asarray(data['y']).astype(np.float)

#for i in range(len(estSize)): 

estimator = RandomForestRegressor(n_estimators=20,max_depth=4) #max features = default

scores = cross_validation.cross_val_score(estimator, mydata, mylabel, cv=10, scoring='mean_squared_error')

mse=np.mean(scores)*-1;

#plt.plot([1,2,3,4], [1,4,9,16], 'ro')