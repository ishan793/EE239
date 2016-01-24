import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np
from sklearn import metrics

def get_model(ip_dim):
	
	model = Sequential()
	model.add(Dense(output_dim = 32, input_dim = ip_dim))
	model.add(Activation('tanh'))
	model.add(Dense(output_dim = 1))
	model.compile(loss='mean_squared_error', optimizer='adam')

	return model

def get_rms(y_true, y_hat):
	err = metrics.mean_squared_error(y_true,y_hat)
	return (err**0.5)

def cross_validate(x,y,n_cv = 10):
	
	n = len(x)
	frac = n/n_cv
	
	for i in range(len(y)):
		y[i] = float(y[i])*100

	
	score = []

	for i in range(n_cv):

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
		print "Training size:",len(x_train),"and testing size:",len(x_test)
		mod = get_model(len(x[0]))
		mod.fit(x_train, y_train, nb_epoch=100, batch_size=8, verbose = 0)
		# y_pred = mod.predict(x_test,batch_size = 8)
		# score.append(get_rms(y_pred,y_test))
		a = mod.evaluate(x_test,y_test,batch_size = 8)
		score.append(a**0.5)
		print "Training complete, score obtained:",score[i]
		

	return score

with open('network.pickle','rb') as f:
	data = pickle.load(f)

x,y = data['x'],data['y']
r = cross_validate(x,y,5)
print r
print float(sum(r))/len(r)

# print y_test[0:10]
# print y_pred[0:10]
# print score