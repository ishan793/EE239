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


def get_selected(all_elem, selected):
    result = []
    for i in selected:
        result.append(all_elem[i])
    return result

data=pickle.load(open("../data/network_data_size.pickle",'rb'))
X=np.array(data['x'],dtype='float')
y=np.array(data['y'],dtype='float')
print X.shape
n_samples=X.shape[0]
y=np.reshape(y,(n_samples,1))
print y.shape

#a=np.hstack((X[:,31:40],X[:,44:45],X[:,51:57]))
a=np.hstack((X[:,31:32],X[:,32:33],X[:,34:36],X[:,51:57]))
X=a


degrees = [1,2,3,4,5]
avg_score=[]
fixed_score=[]
rmse=[]

for i in range(len(degrees)):
    poly=PolynomialFeatures(degree=degrees[i])
    X_trans = poly.fit_transform(X)
    kf=cross_validation.KFold(len(X_trans),10,True)

    for train_index,test_index in kf:
    	X_train = get_selected(X_trans, train_index)
        X_test = get_selected(X_trans, test_index)
        y_train = get_selected(y, train_index)
        y_test = get_selected(y, test_index)

    	regr =LinearRegression()
    	regr.fit(X_train,y_train)
    	y_pred = regr.predict(X_test)    
    	avg_score.append((mean_squared_error(y_test,y_pred)**0.5))	
    rmse.append(np.mean(avg_score))

print rmse
plt.scatter(degrees,rmse)
plt.xlabel('degrees')
plt.ylabel('RMSE')
plt.show()









