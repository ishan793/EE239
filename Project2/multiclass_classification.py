from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier as ovr
from sklearn.multiclass import OneVsOneClassifier as ovo
import pickle
import numpy as np

clf_list = [GaussianNB(), LinearSVC()]
clf_name = ['Gaussian Naive Bayes', 'SVM']

with open('multiclass_db.pickle') as f:
	data = pickle.load(f)

train_data = data['train']
test_data = data['test']

print 'Using One vs All scheme'
for i in range(len(clf_list)):
	print 'Using', str(clf_name[i])
	clf = ovr(clf_list[i])
	clf.fit(train_data['x'], train_data['y'])
	test_pred = clf.predict(test_data['x'])
	print np.mean(test_pred == test_data['y'])
print 'Using One vs One scheme'
for i in range(len(clf_list)):
	print 'Using', str(clf_name[i])
	clf = ovo(clf_list[i])
	clf.fit(train_data['x'], train_data['y'])
	test_pred = clf.predict(test_data['x'])
	print np.mean(test_pred == test_data['y'])