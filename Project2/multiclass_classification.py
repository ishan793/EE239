from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier as ovr
from sklearn.multiclass import OneVsOneClassifier as ovo
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import matplotlib.pyplot as plt
import pickle
import numpy as np

import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)

def plot_confusion_matrix(c_matrix, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(c_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    cat = ['Class 1','Class 2','Class 3','Class 4']
    tick_marks = np.arange(len(cat))
    plt.xticks(tick_marks, cat, rotation=45)
    plt.yticks(tick_marks, cat)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


clf_list = [GaussianNB(), LinearSVC()]
clf_name = ['Gaussian Naive Bayes', 'SVM']
s_name = ['One vs One', 'One vs All']

with open('multiclass_db.pickle') as f:
	data = pickle.load(f)



train_data = data['train']
test_data = data['test']

print 'Using One vs All scheme'
t = "One vs One"
for i in range(len(clf_list)):
	print 'Using', str(clf_name[i])
	clf = ovr(clf_list[i])
	clf.fit(train_data['x'], train_data['y'])
	test_pred = clf.predict(test_data['x'])
	
	print 'Accuracy',np.mean(test_pred == test_data['y'])
	print 'Precision',precision_score(test_data['y'], test_pred)
	print 'Recall',recall_score(test_data['y'], test_pred)
	
	ti = t + " "+clf_name[i]

	cm = confusion_matrix(test_data['y'], test_pred)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.figure()
	plot_confusion_matrix(cm,title = ti)
	plt.savefig(ti)
t = "One vs Rest"
print 'Using One vs One scheme'
for i in range(len(clf_list)):
	print 'Using', str(clf_name[i])
	clf = ovo(clf_list[i])
	clf.fit(train_data['x'], train_data['y'])
	test_pred = clf.predict(test_data['x'])
	
	print 'Accuracy is',np.mean(test_pred == test_data['y'])
	print 'Precision score',precision_score(test_data['y'], test_pred)
	print 'Recall',recall_score(test_data['y'], test_pred)
	
	ti = t + " "+clf_name[i]
	cm = confusion_matrix(test_data['y'], test_pred)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.figure()
	plot_confusion_matrix(cm, title = ti)
	plt.savefig(ti)

