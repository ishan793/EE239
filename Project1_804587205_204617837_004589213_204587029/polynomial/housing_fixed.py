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


data = pickle.load( open("../data/housing_data.pickle", "rb" ) )
X=np.array(data['x'],dtype='float')
y=np.array(data['y'],dtype='float')
print X.shape
n_samples=X.shape[0]
y=np.reshape(y,(n_samples,1))
print y.shape


#For Housing dataset
X_train_fixed=X[50:,:]
y_train_fixed=y[50:,:]
X_test_fixed=X[0:50,:]
y_test_fixed=y[0:50,:]

degrees = [1,2,3,4,5,6]
avg_score=[]
fixed_score=[]

for i in range(len(degrees)):
    poly=PolynomialFeatures(degree=degrees[i])
    X_train_trans=poly.fit_transform(X_train_fixed)
    X_test_trans=poly.fit_transform(X_test_fixed)
    regr=LinearRegression()
    regr.fit(X_train_trans,y_train_fixed)
    y_pred=regr.predict(X_test_trans)
    fixed_score.append((mean_squared_error(y_test_fixed,y_pred)**0.5))
print fixed_score
plt.scatter(degrees,fixed_score)
plt.xlabel('degrees')
plt.ylabel('RMSE')
plt.show()    
