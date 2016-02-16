# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 22:50:10 2016

@author: Tushar Sudhakar Jee
"""

from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import text
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
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
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(clean_data_ct1)


print vectors.shape


svd=TruncatedSVD(n_components=50, n_iter=10,random_state=42)
svd.fit(vectors)
vectors_new=svd.transform(vectors)

print vectors_new.shape
print type(vectors_new)

'''cat=['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware']
train_cat=fetch_20newsgroups(subset='test',categories=cat,shuffle=True,random_state=42,remove=('headers','footers','quotes'))
data_cat=train_cat.data
print len(data_cat),"number of computer class documents"
'''
label0=[0]*2343
label1=[1]*2389
label=label0+label1

#clf = MultinomialNB().fit(vectors_new, label)

clf = linear_model.LinearRegression(vectors_new,label)

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
vectors_test = vectorizer.fit_transform(clean_test)
vectors_test_new=svd.transform(vectors_test)
predicted = clf.predict(vectors_test_new)
#np.mean(predicted == twenty_test.target) 
#1560 computer docu