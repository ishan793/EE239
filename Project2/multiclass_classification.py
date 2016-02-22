from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import pickle
import numpy as np

clf_list = [GaussianNB(), LinearSVC()]
clf_name = ['Gaussian Naive Bayes', 'SVM']

with open('multiclass_db.pickle') as f:
	data = pickle.load(f)

train_data = data['train']
test_data = data['test']

for i in range(len(clf_list)):
	clf = clf_list[i]
	print 'Using', str(clf_name[i]),'Classifier'

	clf.fit(train_data['x'], train_data['y'])

	test_pred = clf.predict(test_data['x'])

	print np.mean(test_pred == test_data['y'])