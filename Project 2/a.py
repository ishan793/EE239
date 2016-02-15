from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import text
import string
stop_words=text.ENGLISH_STOP_WORDS
stemmer2=SnowballStemmer("english")

categories_ct=['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware']
train_ct=fetch_20newsgroups(subset='train',categories=categories_ct,shuffle=True,random_state=42,remove=('headers','footers','quotes'))
data_ct=train_ct.data
clean_data_ct=[]
for i in range(len(data_ct)):
	temp=data_ct[i]

  	#words = temp.translate(unicode.maketrans("",""), unicode.punctuation).lower().split()
	temp=re.sub("[,.-:/()]"," ",temp)
   	temp=temp.lower()
   	words=temp.split()
   	after_stop=[w for w in words if not w in stop_words]
   	after_stem=[stemmer2.stem(plural) for plural in after_stop]
   	temp=" ".join(after_stem)
   	clean_data_ct.append(temp)

# TFIDF vector of the clean data(after removing punctuation , stemming and stop_words)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(clean_data_ct)
print vectors.shape


categories_ra=['rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
train_ra=fetch_20newsgroups(subset='train',categories=categories_ra,shuffle=True,random_state=42,remove=('headers','footers','quotes'))
data_ra=train_ra.data

categories_partc=['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','misc.forsale','soc.religion.christian']

# Counting the number of documents in each class

number=[]
for target in train_ct.target_names:
	number.append(len(train_ct.target))

for target in train_ra.target_names:
	number.append(len(train_ra.target))

