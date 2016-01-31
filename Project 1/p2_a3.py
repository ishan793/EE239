from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import preprocessing as pp
import numpy as np
import pickle
import matplotlib.pyplot as plt
def cross_validate(x,y,n_cv = 10, alpha_ = 0.1):
	'''
	This function used for cross validation. The number of sets is given by
	the user.
	'''
	n = len(x)
	frac = int(n/n_cv)

	score = []
	# create testing and training sets.
	for i in range(n_cv):
		# for this iteration, define the start and end points of test set
		# based on start and end of test, create training set
		start = i*frac
		end = (i+1)*frac

		if end != n:
			x_test,y_test = x[start:end],np.array(y[start:end], dtype = 'float')
			x_train = np.vstack((x[:start],x[end:]))
			y_train = np.array(y[:start] + y[end:], dtype = 'float')
		else:
			x_test, y_test =  x[start:], np.array(y[start:], dtype = 'float')
			x_train = x[:start]
			y_train = np.array(y[:start], dtype = 'float')
		
		# create and train model			
		# print "Training size:",len(x_train),"and testing size:",len(x_test)
		regr = Lasso(alpha = alpha_)
		regr.fit(x_train,y_train)
		# y_pred = regr.predict(x_test)
		# score.append(np.mean(y_pred - y_test)**2)
		score.append(regr.score(x_test,y_test))
		# print "Training complete, score obtained:",score[i]
		
	return score

with open('housing_data.pickle','rb') as f:
	data = pickle.load(f)

x,y = data['x'],data['y']
x = np.array(x, dtype = 'float')
x = pp.normalize(x)

# print type(x[0])
n = 10
x_axis = []
y_axis = []
val = 0.0001
step = 0.01
for i in range(100):
	val = val + step
	a = cross_validate(x,y,alpha_ = val)
	y_axis.append(np.mean(a))
	x_axis.append(val)
	

plt.figure()
plt.plot(x_axis,y_axis)
plt.show()
	
