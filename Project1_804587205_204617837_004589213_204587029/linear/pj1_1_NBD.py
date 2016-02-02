'''
Generates actual labels(y),fitted values(yp) and residuals(y-yp). 
Also includes the plot for residual(y-yp) versus fitted values(yp)
NETWORK BACKUP DATASET
'''

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.feature_selection import RFE
import numpy as np
import pickle
import csv

data = pickle.load(open("../data/network_data_size.pickle", 'r+'))
x = np.array(data['x'], dtype=np.float)
y = np.array(data['y'], dtype=np.float)

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(x, y, test_size=0.1,

                                                                                             random_state=0)

reg = linear_model.LinearRegression()

rfe = RFE(reg, n_features_to_select=32)

rfe.fit(x, y)
yp = rfe.predict(x)

#reg.fit(x,y)
#yp=reg.predict(x)

residuals= y-yp
print ('list of residual values '),residuals


print ('predicted values are as '),yp

print ('number of features selected \n'), rfe.n_features_
print ('regression coefficient rankings are as \n'), rfe.ranking_
print ('mask of features is as \n'), rfe.support_

j = []
for o, q in enumerate(rfe.ranking_):
    if (q == 1):
        j.append(o)

print('indexes of selected features \n'), j
#scores = cross_validation.cross_val_score(reg, x, y, scoring='mean_squared_error', cv=10)

scores = cross_validation.cross_val_score(rfe, x, y, scoring='mean_squared_error', cv=10)

print ("RMSE is as "), (abs(scores) ** 0.5).mean()

#residuals v/s fitted values plot:
plt.scatter(yp,residuals, c="g")
plt.xlabel('Predicted labels')
plt.ylabel('Residual values')



plt.show()
