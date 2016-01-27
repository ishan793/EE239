
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
import pickle
from sklearn.metrics import mean_squared_error

#Entire data set is loaded. Then we split the data in 10 folds 
# data=pickle.load(open('network.pickle','rb'))
# X=np.array(data['x'],dtype='float')
# y=np.array(data['y'],dtype='float')
# n_samples=X.shape[0]
# y=np.reshape(y,(n_samples,1))
# features_train,features_test,labels_train,labels_test=cross_validation.train_test_split(X,y,test_size=0.1,random_state=0)

#Load the data according to workflow.And fit a linear regression model to it.
#Workflow 1
'''wf_base_name = 'network_wf'
res = []
a = 0
count = 0
for i in range(5):
	cur_name = wf_base_name + str(i+1)+'.pickle'
	data_wf = pickle.load(open(cur_name))
	
	x_wf = np.array(data_wf['x'],dtype='float')
	y_wf = np.array(data_wf['y'],dtype='float')
	n_samples_wf = x_wf.shape[0]
	count += n_samples_wf
	y_wf = np.reshape(y_wf,(n_samples_wf,1))

	n_test = int(0.1*n_samples_wf)
	x_wf_train,x_wf_test = x_wf[n_test:], x_wf[:n_test]
	y_wf_train, y_wf_test = y_wf[n_test:], y_wf[:n_test]

	reg_wf = linear_model.LinearRegression(fit_intercept=True,normalize=True)
	reg_wf.fit(x_wf_train,y_wf_train)
	y_pred = reg_wf.predict(x_wf_test)
	a += sum(((y_pred - y_wf_test)**2))
	
	# scores_wf =cross_validation.cross_val_score(reg_wf,
	#         X_wf, y_wf, scoring = "mean_squared_error", cv=10)

print (a/count)**0.5'''

def find_wf(sample):
	if sample[28]==1:
		return 1
	elif sample[29]==1:
		return 2
	elif sample[30]==1:
		return 3
	elif sample[31]==1:
		return 4
	else:
		return 5				


def split_train1(x_train,y_train):
	split_train = {'x':{},'y':{}}
	for i in range(x_train.shape[0]):
		cur_wf = find_wf(x_train[i])
		if cur_wf not in split_train['x']:
			split_train['x'][cur_wf] = []
			split_train['y'][cur_wf] = []
		split_train['x'][cur_wf].append(x_train[i])
		split_train['y'][cur_wf].append(y_train[i])
 	return split_train

def train_models(split_train):
	trained_models = {}
	mod_ind = split_train['x'].keys()
	for i in mod_ind:
		x_train = split_train['x'][i]
		y_train = split_train['y'][i]
		mod = linear_model.LinearRegression()
		mod.fit(x_train,y_train)
		trained_models[i] = mod
	return trained_models

def test_data(split_test, trained_models):
	mod_ind = split_test['x'].keys()
	res = []
	a=0
	n_samples=0
	for i in mod_ind:
		x_test = split_test['x'][i]
		y_pred = trained_models[i].predict(x_test)
		n_samples+=len(x_test)
		a += sum(((y_pred - split_test['y'][i])**2))
	print (a/n_samples)**0.5	

def load1():
 	data=pickle.load(open('network.pickle'))
 	x=np.array(data['x'],dtype='float')
	y=np.array(data['y'],dtype='float')
	n_samples=x.shape[0]
	y=np.reshape(y,(n_samples,1))
	features_train,features_test,labels_train,labels_test=cross_validation.train_test_split(x,y,test_size=0.1,random_state=0)
 	split_train=split_train1(features_train,labels_train)
 	trained_models=train_models(split_train)
 	#print trained_models[1].coef_
 	split_test=split_train1(features_test,labels_test)
 	test_data(split_test,trained_models)

load1()



# #Workflow 2
# data_wf2=pickle.load(open('network_wf2.pickle','rb'))
# X_wf2=np.array(data_wf2['x'],dtype='float')
# y_wf2=np.array(data_wf2['y'],dtype='float')
# n_samples_wf2=X_wf2.shape[0]
# y_wf2=np.reshape(y_wf2,(n_samples_wf2,1))

# reg_wf2=linear_model.LinearRegression(fit_intercept=True,normalize=True)
# # lreg_wf2=reg_wf2.fit(X_wf2,y_wf2)
# #labels_wf2=reg_wf2.predict()
# scores_wf2=cross_validation.cross_val_score(reg_wf2,
#         X_wf2, y_wf2, scoring="mean_squared_error", cv=10)
# avg_wf2=(abs(scores_wf2)**0.5).mean()

# #Workflow3
# data_wf3=pickle.load(open('network_wf3.pickle','rb'))
# X_wf3=np.array(data_wf3['x'],dtype='float')
# y_wf3=np.array(data_wf3['y'],dtype='float')
# n_samples_wf3=X_wf3.shape[0]
# y_wf3=np.reshape(y_wf3,(n_samples_wf3,1))

# reg_wf3=linear_model.LinearRegression(fit_intercept=True,normalize=True)
# # lreg_wf3=reg_wf3.fit(X_wf3,y_wf3)
# #labels_wf3=reg_wf3.predict()
# scores_wf3=cross_validation.cross_val_score(reg_wf3,
#         X_wf3, y_wf3, scoring="mean_squared_error", cv=10)
# avg_wf3=(abs(scores_wf3)**0.5).mean()

# #Workflow4
# data_wf4=pickle.load(open('network_wf4.pickle','rb'))
# X_wf4=np.array(data_wf4['x'],dtype='float')
# y_wf4=np.array(data_wf4['y'],dtype='float')
# n_samples_wf4=X_wf4.shape[0]
# y_wf4=np.reshape(y_wf4,(n_samples_wf4,1))

# reg_wf4=linear_model.LinearRegression(fit_intercept=True,normalize=True)
# # lreg_wf4=reg_wf4.fit(X_wf4,y_wf4)
# #labels_wf4=reg_wf4.predict()
# scores_wf4=cross_validation.cross_val_score(reg_wf4,
#         X_wf4, y_wf4, scoring="mean_squared_error", cv=10)
# avg_wf4=(abs(scores_wf4)**0.5).mean()


# #Workflow5
# data_wf5=pickle.load(open('network_wf5.pickle','rb'))
# X_wf5=np.array(data_wf5['x'],dtype='float')
# y_wf5=np.array(data_wf5['y'],dtype='float')
# n_samples_wf5=X_wf5.shape[0]
# y_wf5=np.reshape(y_wf5,(n_samples_wf5,1))

# reg_wf5=linear_model.LinearRegression(fit_intercept=True,normalize=True)
# lreg_wf5=reg_wf5.fit(X_wf5,y_wf5)
# labels_wf5=reg_wf5.predict(X_wf5)
# print ((labels_wf5-y_wf5)**2).mean()
# scores_wf5=cross_validation.cross_val_score(reg_wf5,
#         X_wf5, y_wf5, scoring="mean_squared_error", cv=10)
# avg_wf5=(abs(scores_wf5)**0.5).mean()
# avg_mse=(avg_wf5+avg_wf4+avg_wf3+avg_wf2+avg_wf1)/5
# print avg_mse
