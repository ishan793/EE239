import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
import numpy as np
from sklearn import preprocessing as pp
import matplotlib.pyplot as plt


def get_model(ip_dim, n_nodes = 32, _lr = 0.001):
	'''
	This function returns the model of a regression model using 
	neural network. It consists of single hidden layer with 
	32 nodes. The activation function for
	each node is tanh. As we are working with regression, the loss function
	used is mean squared error and the optimizer used is Adam.
	'''	
	model = Sequential()
	model.add(Dense(output_dim = n_nodes, input_dim = ip_dim))
	model.add(Activation('tanh'))
	# model.dropout()
	# model.add(Dense(output_dim = 16, input_dim = ip_dim))
	# model.add(Activation('tanh'))
	model.add(Dense(output_dim = 1))
	opt = Adam(lr = _lr)
	model.compile(loss='mean_squared_error', optimizer = opt)

	return model

def cross_validate(x,y,n_cv = 10, mf = 100, lr = 0.01, nodes = 32):
	'''
	This function used for cross validation. The number of sets is given by
	the user. Based on parameter optimization for this data set, the learning 
	rate is set as 0.01 and n_nodes as 32. Similar arguments are used for number of
	epochs.
	'''
	n = len(x)
	frac = int(n/n_cv)
	
	for i in range(len(y)):
		y[i] = float(y[i])*mf

	score = []
	# create testing and training sets.
	for i in range(n_cv):
		# for this iteration, define the start and end points of test set
		# based on start and end of test, create training set
		start = i*frac
		end = (i+1)*frac

		if end != n:
			x_test,y_test = x[start:end],np.array(y[start:end])
			x_train = np.vstack((x[:start],x[end:]))
			y_train = np.array(y[:start] + y[end:])
		else:
			x_test, y_test =  x[start:], np.array(y[start:])
			x_train = x[:start]
			y_train = np.array(y[:start])
		
		# create and train model			
		print "Training size:",len(x_train),"and testing size:",len(x_test)
		mod = get_model(len(x[0]), _lr = lr, n_nodes = nodes)
		mod.fit(x_train, y_train, nb_epoch=100, batch_size=8, verbose = 0)
		# evaluate the model
		a = mod.evaluate(x_test,y_test,batch_size = 8)
		rmse = (a**0.5)/mf
		score.append(rmse)
		print "Training complete, score obtained:",score[i]
		
	return score

def lr_plot(x,y,mf):
	'''
	Function to plot variation of RMSE with varying learning rate
	Range of learning rate considered : 10^-6 - 1
	'''
	n_train = int(0.9*len(x))

	for i in range(len(y)):
		y[i] = float(y[i])*mf

	x_train, x_test = x[:n_train], x[n_train:]
	y_train, y_test = y[:n_train], y[n_train:]


	lr_ = []
	res = []

	print "Data created"
	print "Training size:",len(x_train),"testing size:",len(x_test)
	for i in range(7):
		lr = 10**(-6+i)
		mod = get_model(len(x[0]), _lr = lr)
		mod.fit(x_train, y_train, nb_epoch=100, batch_size=8, verbose = 0)
		# evaluate the model
		a = mod.evaluate(x_test,y_test,batch_size = 8)
		rmse = (a**0.5)/mf
		print "current lr is:",lr,"and error is:",rmse
		lr_.append(lr)
		res.append(rmse)

	plt.figure()
	plt.semilogx(lr_,res)
	plt.xlabel('Learning rate')
	plt.ylabel('RMSE')
	plt.title('RMSE vs Learning rate')
	plt.show()
		

def nodes_plot(x,y):
	'''
	Function to plot variation of RMSE with varying number of nodes
	in the middle layer
	'''
	n_train = int(0.9*len(x))
	mf = 100
	for i in range(len(y)):
		y[i] = float(y[i])*mf

	x_train, x_test = x[:n_train], x[n_train:]
	y_train, y_test = y[:n_train], y[n_train:]

	n = 10
	n_ = []
	res = []

	print "Data created"
	print "Training size:",len(x_train),"testing size:",len(x_test)
	for i in range(10):
	
		mod = get_model(len(x[0]), n_nodes = n)
		mod.fit(x_train, y_train, nb_epoch=100, batch_size=8, verbose = 0)
		# evaluate the model
		a = mod.evaluate(x_test,y_test,batch_size = 8)
		rmse = (a**0.5)/mf
		print "current n_nodes is:",n,"and error is:",rmse
		n_.append(n)
		res.append(rmse)
		n += 10

	plt.figure()
	plt.plot(n_,res)
	plt.xlabel('Number of nodes')
	plt.ylabel('RMSE')
	plt.title('RMSE vs Number of nodes')
	plt.show()	


with open('../data/network_data_size.pickle','rb') as f:
	data = pickle.load(f)

x,y = data['x'],data['y']
x = pp.normalize(x)

# lr_plot(x,y,100)
# nodes_plot(x,y)

res = cross_validate(x,y)
print float(sum(res)/len(res))
