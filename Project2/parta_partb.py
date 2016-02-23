from sklearn.datasets import fetch_20newsgroups
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import text
import numpy as np
#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

stop_words=text.ENGLISH_STOP_WORDS
stemmer2=SnowballStemmer("english")
cat_ct=['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware']
cat_ra=['rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
categories_ct=['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
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

train_ra=fetch_20newsgroups(subset='train',categories=cat_ra,shuffle=True,random_state=42,remove=('headers','footers','quotes'))
data_ra=train_ra.data

print 'number of documents in Recreational Activity are',len(data_ra)
print 'number of documents in Computer Technology are',(len(data_ct)-len(data_ra))


frequencies = [480, 584, 591, 590, 578, 593, 585, 594, 598, 597, 600, 595, 591, 594, 593, 599, 546, 564, 465, 377]

#pos = np.arange(len(alphab))
pos=np.arange(20)
width = 1.0     # gives histogram aspect to the bar diagram

ax = plt.axes()
ax.set_xticks(pos + (width))
#ax.set_xticklabels(alphab)
plt.xlabel('classes')
plt.ylabel('Number of documents')
plt.bar(pos, frequencies, width, color='r')
plt.savefig('histogram.png')


print 'final number of terms extracted are ',vectors.shape[1]   


