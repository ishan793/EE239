from sklearn.datasets import fetch_20newsgroups
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import text
stop_words=text.ENGLISH_STOP_WORDS
stemmer2=SnowballStemmer("english")

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

'''
categories_ra=[]
train_ra=fetch_20newsgroups(subset='train',categories=categories_ra,shuffle=True,random_state=42,remove=('headers','footers','quotes'))
data_ra=train_ra.data
categories_partc=['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','misc.forsale','soc.religion.christian']
'''


# Counting the number of documents in each class
'''all_cats = list(fetch_20newsgroups(subset='train').target_names)
final_res = []
for cat in all_cats:
    train_cat = fetch_20newsgroups(subset='train',categories=[cat],remove=('headers','footers','quotes'))
    data_cat = train_cat.data
    res_cat = ''
    for doc in data_cat:
        res_cat += ' '+doc
    final_res.append(res_cat)
'''     


