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
	each node is tanh. As we are working with regression, final layer has linear activation
	function and the loss function used is mean squared error and the optimizer used is Adam.
	'''	
	model = Sequential()
	model.add(Dense(output_dim = n_nodes, input_dim = ip_dim))
	model.add(Activation('tanh'))

	model.add(Dense(output_dim = 1))
	model.add(Activation('linear'))
	opt = Adam(lr = _lr)
	model.compile(loss='mean_squared_error', optimizer = opt)

	return model

def cross_validate(x,y,n_cv = 10, mf = 100, lr = 0.01):
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
	print "Running",n_cv,"fold cross validation, train size",(n-frac),"and testing size",frac
	print "Number of nodes:",nodes,"and learning rate:",lr
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

def lr_plot(x,y,file_name):
	'''
	Function to plot variation of RMSE with varying learning rate
	Range of learning rate considered : 10^-6 - 1
	'''
	n_train = int(0.9*len(x))
	mf = 100
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
	# plt.show()
	plt.savefig(file_name)
		

def nodes_plot(x,y,file_name, flag = 1):
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
	if flag != 1:
		n = 2
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
		if flag == 1:
			n += 10
		else:
			n += 1

	plt.figure()
	plt.plot(n_,res)
	plt.xlabel('Number of nodes')
	plt.ylabel('RMSE')
	plt.title('RMSE vs Number of nodes')
	plt.savefig(file_name)
	# plt.show()	

def parameter_plot(p_type, data):
	if data == 'size':
		f_name = '../data/network_data_size.pickle'
		s_name = 'size'
		f = 1
	elif data == 'time':
		f_name = '../data/network_backup_time.pickle'
		s_name = 'time'
		f = 1
	elif data == 'boston':
		s_name = 'housing'
		f_name = '../data/housing_data.pickle'
		f = 2

	with open(f_name,'rb') as f:
		data = pickle.load(f)
	
	x,y = data['x'],data['y']
	x = pp.normalize(x)
	
	if p_type == 'lr':
		print "Generating plot for learning rate"
		s_name += 'lr_rmse.png'
		lr_plot(x,y,s_name)
	
	elif p_type == 'nodes':
		print "Generating plot for number of nodes"
		s_name += 'nodes_rmse.png'
		nodes_plot(x,y,s_name,flag = f)

def get_res(data):
	l = 0.01
	if data == 'size':
		print "Running cross validation for Data size"
		f_name = '../data/network_data_size.pickle'
		n_nodes = 32
	elif data == 'time':
		print "Running cross validation for Backup time"
		f_name = '../data/network_backup_time.pickle'
		n_nodes = 60
	elif data == 'boston':
		print "Running cross validation for Boston dataset"
		f_name = '../data/housing_data.pickle'
		n_nodes = 11
		l = 0.1

	with open(f_name,'rb') as f:
		data = pickle.load(f)
	
	x,y = data['x'],data['y']
	x = pp.normalize(x)
	
	res = cross_validate(x,y,lr = l,nodes = n_nodes)
	print float(sum(res)/len(res))

# parameter_plot(p_type = 'nodes',data = 'boston')
# parameter_plot(p_type = 'lr',data = 'boston')

get_res('time')	
get_res('size')
get_res('boston')


