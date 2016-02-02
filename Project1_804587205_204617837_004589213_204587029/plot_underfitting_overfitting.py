"""
============================
Underfitting vs. Overfitting
============================

This example demonstrates the problems of underfitting and overfitting and
how we can use linear regression with polynomial features to approximate
nonlinear functions. The plot shows the function that we want to approximate,
which is a part of the cosine function. In addition, the samples from the
real function and the approximations of different models are displayed. The
models have polynomial features of different degrees. We can see that a
linear function (polynomial with degree 1) is not sufficient to fit the
training samples. This is called **underfitting**. A polynomial of degree 4
approximates the true function almost perfectly. However, for higher degrees
the model will **overfit** the training data, i.e. it learns the noise of the
training data.
We evaluate quantitatively **overfitting** / **underfitting** by using
cross-validation. We calculate the mean squared error (MSE) on the validation
set, the higher, the less likely the model generalizes correctly from the
training data.
"""

#print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
np.random.seed(0)

#data = pickle.load( open( "data/housing_data.pickle", "rb" ) )
data=pickle.load(open('data/network_data_size.pickle','rb'))
X=np.array(data['x'],dtype='float')
y=np.array(data['y'],dtype='float')
print X.shape
n_samples=X.shape[0]
y=np.reshape(y,(n_samples,1))
print y.shape
degrees = [2]
avg_score=[]
fixed_score=[]
#length_of_32 [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
#length_of_16 [31, 32, 33, 34, 35, 36, 37, 38, 39, 44, 51, 52, 53, 54, 55, 56]
#length_of_10 [31, 32, 34, 35, 51, 52, 53, 54, 55, 56]
'''reg = linear_model.LinearRegression()
rfe = RFE(reg,n_features_to_select=16)
rfe.fit(X, y)
#print ('number of features selected \n'), rfe.n_features_
#print ('regression coefficient rankings are as \n'), rfe.ranking_
#print ('mask of features is as \n'), rfe.support_

j = []
for o,q in enumerate(rfe.ranking_):
    if ( q==1 ):
        j.append(o)

print('indexes of selected features \n'), j
'''


#a=np.hstack((X[:,31:40],X[:,44:45],X[:,51:57]))
a=np.hstack((X[:,31:32],X[:,32:33],X[:,34:36],X[:,51:57]))
X=a

X_test=X[0:50,:]
y_test=y[0:50,:]
X_train=X[51:,:]
y_train=y[51:,:]
#plt.figure(figsize=(14, 5))
'''
n = len(X)
n_cv=10
frac = int(n/n_cv)

for i in range(n_cv):
        # for this iteration, define the start and end points of test set
        # based on start and end of test, create training set
        start = i*frac
        end = (i+1)*frac

        if end != n:
            x_test,y_test = X[start:end],np.array(y[start:end])
            x_train = np.vstack((X[:start],X[end:]))
            y_train = np.vstack((y[:start],y[end:]))
        else:
            x_test, y_test =  X[start:], np.array(y[start:])
            x_train = X[:start]
            y_train = np.array(y[:start])
        
        # create and train model            
        
        print "Training size:",len(x_train),"and testing size:",len(x_test)

        linear_regression=LinearRegression()
        linear_regression.fit(x_train,y_train)
        y_pred = linear_regression.predict(x_test)    
        fixed_score.append((mean_squared_error(y_test,y_pred)**0.5))
#scores = cross_validation.cross_val_score(linear_regression,
 #       X, y, scoring="mean_squared_error", cv=10)
#scores=np.average((abs(scores)**0.5))
#avg_score.append(scores)
#print avg_score
print np.mean(fixed_score)
'''
'''linear_regression=LinearRegression()
linear_regression.fit(X,y)
y_pred = linear_regression.predict(X)    
fixed_score.append((mean_squared_error(y,y_pred)**0.5))
print fixed_score
'''

for i in range(len(degrees)):
    polynomial_features = PolynomialFeatures(degree=degrees[i],interaction_only=True,
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    #pipeline.fit(X,y)

    # Evaluate the models using crossvalidation
    scores = cross_validation.cross_val_score(pipeline,
        X, y, scoring="mean_squared_error", cv=10)
    scores=np.average((abs(scores)**0.5))
    avg_score.append(scores)

print avg_score
plt.scatter(degrees,avg_score)
plt.show()


'''plt.figure(figsize=(14,5))
for i in range(len(degrees)):
    ax=plt.subplot(1,len(degrees),i+1)
    plt.setp(ax,xticks=(),yticks=())

    poly=PolynomialFeatures(degree=degrees[i])
    X_train_trans = poly.fit_transform(X_train)
    X_test_trans = poly.fit_transform(X_test)
    regr =LinearRegression()
    regr.fit(X_train_trans,y_train)
    y_pred = regr.predict(X_test_trans)    
    fixed_score.append((mean_squared_error(y_test,y_pred)**0.5))
    #plt.plot(range(len(y_test)),(y_test-pipeline.predict(X_test)),range(len(y_test)),[0]*len(y_test))


print fixed_score
plt.scatter(degrees,fixed_score)
plt.show()
'''