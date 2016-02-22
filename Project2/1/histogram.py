from sklearn.datasets import fetch_20newsgroups as ng20

cat = ['comp.graphics']
nws_train = ng20(subset='train', categories = cat)

