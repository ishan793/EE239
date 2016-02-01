from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso, LinearRegression
from sklearn import preprocessing as pp
import numpy as np
import pickle
import matplotlib.pyplot as plt

def cross_validate(x,y,n_cv = 10, alpha_ = 0.1, mod = 'ridge'):
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
		if mod == 'ridge':
			# print 'Using ridge regression'
			regr = Ridge(alpha = alpha_)
		elif mod == 'lasso':
			# print 'Using Lasso regression'
			regr = Lasso(alpha = alpha_)
		else:
			regr = LinearRegression()

		regr.fit(x_train,y_train)
		y_pred = regr.predict(x_test)
		temp = (y_pred - y_test)**2
		score.append((np.mean(temp))**0.5)
		
	return np.mean(score)

def find_alpha(alpha,model = 'ridge'):

	with open('../data/housing_data.pickle','rb') as f:
		data = pickle.load(f)

	x,y = data['x'],data['y']
	x = np.array(x, dtype = 'float')


	x_axis = []
	y_axis = []

	min_alpha = 0
	min_val = -1

	for i in alpha:
		# print i
		val = i
		a = cross_validate(x,y,alpha_ = val, mod = model)
		y_axis.append(a)
		if min_val == -1:
			min_val = a
			min_alpha = i
		if min_val > a:
			min_val = a
			min_alpha = i
	
	print 'For',model,'Minimum score is',str(min_val),'at alpha = ',(min_alpha)
	
	if len(alpha) != 1:		
		plt.figure()
		plt.plot(alpha,y_axis)
		plt.xlabel('Alpha')
		plt.ylabel('RMSE')
		plt.title(str(model)+' RMSE vs Alpha')
		# plt.show()
		s_name = model+'_rmse_alpha.png'
		plt.savefig(s_name)

a_lasso = [(i+1)*0.05 for i in range(20)]
a_ridge = range(100)

# a = [10**(-5+i) for i in range(5)] + range(1,5)
# print a
a = [0.001,0.01,0.1]

find_alpha(a_ridge,'ridge')
find_alpha(a_lasso,'lasso')
