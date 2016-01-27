from sklearn.linear_model import RidgeCV

with open('network.pickle','rb') as f:
	data = pickle.load(f)

x,y = data['x'],data['y']

regr = RidgeCV(alphas = [0])
