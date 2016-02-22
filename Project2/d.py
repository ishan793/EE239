# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 22:50:10 2016

@author: Tushar Sudhakar Jee
"""
import cPickle,gzip
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import text
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

stop_words=text.ENGLISH_STOP_WORDS
stemmer2=SnowballStemmer("english")

categories_ct=['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
train_ct1=fetch_20newsgroups(subset='train',categories=categories_ct,shuffle=True,random_state=42,remove=('headers','footers','quotes'))
data_ct1=train_ct1.data
clean_data_ct1=[]

for i in range(len(data_ct1)):
	temp=data_ct1[i]
  	#words = temp.translate(unicode.maketrans("",""), unicode.punctuation).lower().split()
	temp=re.sub("[,.-:/()]"," ",temp)
   	temp=temp.lower()
   	words=temp.split()
   	after_stop=[w for w in words if not w in stop_words]
   	after_stem=[stemmer2.stem(plural) for plural in after_stop]
   	temp=" ".join(after_stem)
   	clean_data_ct1.append(temp)

# TFIDF vector of the clean data(after removing punctuation , stemming and stop_words)
# vectorizer = TfidfVectorizer()
# vectors = vectorizer.fit_transform(clean_data_ct1)
# create bow repr

bow_vec = CountVectorizer(max_df = 0.75)
tfidf_vec = TfidfTransformer()

bow_vec.fit(clean_data_ct1)

train_doc = bow_vec.transform(clean_data_ct1)
print train_doc.shape

tfidf_vec.fit(train_doc)
train_doc = tfidf_vec.transform(train_doc)
print train_doc.shape


svd=TruncatedSVD(n_components=50, n_iter=10,random_state=42)
svd.fit(train_doc)
vectors_new=svd.transform(train_doc)

print vectors_new.shape
print type(vectors_new)

'''#cat=['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware']
cat=['rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
train_cat=fetch_20newsgroups(subset='test',categories=cat,shuffle=True,random_state=42,remove=('headers','footers','quotes'))
data_cat=train_cat.data
print len(data_cat),"number of computer class documents"
'''
temp = train_ct1.target
label = []
for i in temp: 
  label.append(int(i<4))


with gzip.open('train_data.pkl.gz','wb') as f:
  cPickle.dump((vectors_new,label),f)
f.close()    


# label0=[0]*2343
# label1=[1]*2389
# label=label0+label1

#clf = MultinomialNB().fit(vectors_new, label)

clf = GaussianNB()
clf.fit(vectors_new,label)

#testing dataset
twenty_test = fetch_20newsgroups(subset='test',
    categories=categories_ct, shuffle=True, random_state=42)
docs_test = twenty_test.data
clean_test=[]
for i in range(len(docs_test)):
	temp=docs_test[i]

  	#words = temp.translate(unicode.maketrans("",""), unicode.punctuation).lower().split()
	temp=re.sub("[,.-:/()]"," ",temp)
   	temp=temp.lower()
   	words=temp.split()
   	after_stop=[w for w in words if not w in stop_words]
   	after_stem=[stemmer2.stem(plural) for plural in after_stop]
   	temp=" ".join(after_stem)
   	clean_test.append(temp)

# TFIDF vector of the clean data(after removing punctuation , stemming and stop_words)
test_doc = bow_vec.transform(clean_test)
print test_doc.shape

test_doc = tfidf_vec.transform(test_doc)
print test_doc.shape


#svd.fit(vectors_test)
vectors_test_new=svd.transform(test_doc)
predicted = clf.predict(vectors_test_new)

y_test_lab = twenty_test.target
label_test = []
for i in y_test_lab: 
  label_test.append(int(i<4))

with gzip.open('test_data.pkl.gz','wb') as f:
  cPickle.dump((vectors_test_new,label_test),f)
f.close()    


print np.mean(predicted == label_test) 
print confusion_matrix(label_test, predicted)
print precision_score(label_test, predicted)
print recall_score(label_test, predicted)