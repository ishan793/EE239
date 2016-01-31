# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:03:42 2016

@author: shubham
"""
import cPickle as pickle
data = pickle.load(open("housing_data.pickle","rb"))
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

mydata = np.asarray(data['x']).astype(np.float)
mylabel = np.asarray(data['y']).astype(np.float)
mydata_normalized = preprocessing.normalize(mydata, norm='l2')
#------------------------------------------------------------------------------

#code for fine-tuning tree depth 
#depth =[4,5,6,7,8,9,10,11,12,13,14]
#mse = [];
#    
#for i in range(len(depth)): 
#
#    estimator = RandomForestRegressor(n_estimators=20,max_depth=depth[i]) #max features = default
#    
#    scores = cross_validation.cross_val_score(estimator, mydata_normalized, mylabel, cv=10, scoring='mean_squared_error')
#
#    mse.append((np.mean(scores)*-1)**(0.5));
#
#plt.plot(depth, mse, 'ro')
#plt.xlabel('depth')
#plt.ylabel('root mean square error')

#------------------------------------------------------------------------------

 #code to find the optimum estimaters for prediction
estm =[20,60,100,140,180,220]
mse = [];
    
for j in range(len(estm)): 

    estimator = RandomForestRegressor(n_estimators=estm[j],max_depth=12) #max features = default

    scores = cross_validation.cross_val_score(estimator, mydata_normalized, mylabel, cv=10, scoring='mean_squared_error')

    mse.append((np.mean(scores)*-1)**(0.5));

#plt.ylim([0.0090,1])
plt.plot(estm, mse)
plt.xlabel('Number of estimaters')
plt.ylabel('root mean square error')