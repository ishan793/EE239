# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:51:05 2016

@author: shubham
"""
from sklearn.datasets import fetch_20newsgroups
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import text
import numpy as np
stop_words=text.ENGLISH_STOP_WORDS
stemmer2=SnowballStemmer("english")

def top_tfidf_feats(row,features, top_n=10):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    print top_feats
    print '\n'
    return




categories_ra=['rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
train_ra=fetch_20newsgroups(subset='train',categories=categories_ra,shuffle=True,random_state=42,remove=('headers','footers','quotes'))
data_ra=train_ra.data
categories_partc=['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','misc.forsale','soc.religion.christian']

# Counting the number of documents in each class
all_cats = list(fetch_20newsgroups(subset='train').target_names)
final_res = []
for cat in all_cats:
    train_cat = fetch_20newsgroups(subset='train',categories=[cat],remove=('headers','footers','quotes'))
    data_cat = train_cat.data
    res_cat = ''
    for doc in data_cat:
        res_cat += ' '+doc
    final_res.append(res_cat)

clean_data=[]
for i in final_res:
	temp = i
  	#words = temp.translate(unicode.maketrans("",""), unicode.punctuation).lower().split()
	temp=re.sub("[,.-:/()?{}*$#&]"," ",temp)
   	temp=temp.lower()
   	words=temp.split()
   	after_stop=[w for w in words if not w in stop_words]
   	after_stem=[stemmer2.stem(plural) for plural in after_stop]
   	temp=" ".join(after_stem)
   	clean_data.append(temp)

# TFIDF vector of the clean data(after removing punctuation , stemming and stop_words)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(clean_data)

row1 = np.squeeze(vectors[3].toarray())
row2 = np.squeeze(vectors[4].toarray())
row3 = np.squeeze(vectors[6].toarray())
row4 = np.squeeze(vectors[15].toarray())

features = vectorizer.get_feature_names()

top_tfidf_feats(row1,features,10)

top_tfidf_feats(row2,features,10)

top_tfidf_feats(row3,features,10)

top_tfidf_feats(row4,features,10)
