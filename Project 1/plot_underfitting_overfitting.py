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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
import pickle
from sklearn.metrics import mean_squared_error
np.random.seed(0)

#data = pickle.load( open( "housing_data.pickle", "rb" ) )
data=pickle.load(open('network.pickle','rb'))
X=np.array(data['x'],dtype='float')
y=np.array(data['y'],dtype='float')
print X.shape
n_samples=X.shape[0]
y=np.reshape(y,(n_samples,1))
print y.shape
degrees = [2,3,4]
avg_score=[]
fixed_score=[]

X_test=X[0:50,:]
y_test=y[0:50,:]
X_train=X[51:,:]
y_train=y[51:,:]
#plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    #ax = plt.subplot(1, len(degrees), i + 1)
    #plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i],
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
   
    #plt.plot(X_test, true_fun(X_test), label="True function")
    #plt.scatter(X, y, label="Samples")
    #plt.xlabel("x")
    #plt.ylabel("y")
    #plt.xlim((0, 1))
    #plt.ylim((-2, 2))
    #plt.legend(loc="best")
    #plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        #degrees[i], -scores.mean(), scores.std()))
#plt.show()'''
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
'''

'''print fixed_score
plt.scatter(degrees,fixed_score)
plt.show()'''
