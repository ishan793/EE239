import pickle
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

stop_words=text.ENGLISH_STOP_WORDS
stemmer2=SnowballStemmer("english")

# categories_ct=['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
cat = ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','misc.forsale','soc.religion.christian']

train_raw_docs = fetch_20newsgroups(subset = 'train',categories = cat,
  shuffle = True,random_state = 42, remove=('headers','footers','quotes'))
train_data_raw = train_raw_docs.data

train_data_clean = []

for temp in train_data_raw:
  	#words = temp.translate(unicode.maketrans("",""), unicode.punctuation).lower().split()
	temp = re.sub("[,.-:/()]"," ",temp)
 	temp = temp.lower()
 	words = temp.split()
 	after_stop = [w for w in words if not w in stop_words]
 	after_stem = [stemmer2.stem(plural) for plural in after_stop]
 	temp = " ".join(after_stem)
 	train_data_clean.append(temp)


bow_vec = CountVectorizer(max_df = 0.75)
tfidf_vec = TfidfTransformer()

bow_vec.fit(train_data_clean)

train_doc = bow_vec.transform(train_data_clean)
print train_doc.shape

tfidf_vec.fit(train_doc)
train_doc = tfidf_vec.transform(train_doc)
print train_doc.shape


svd = TruncatedSVD(n_components=50, n_iter=10,random_state=42)
svd.fit(train_doc)

lsi_train = svd.transform(train_doc)
print 'Dimensions of training data are:',str(lsi_train.shape)
labels_train = train_raw_docs.target

#testing dataset
twenty_test = fetch_20newsgroups(subset='test',
    categories = cat, shuffle = True, random_state = 42, remove=('headers','footers','quotes'))

docs_test = twenty_test.data
clean_test=[]
for temp in docs_test:

	temp = re.sub("[,.-:/()]"," ",temp)
 	temp = temp.lower()

 	words = temp.split()
 	after_stop = [w for w in words if not w in stop_words]
 	after_stem = [stemmer2.stem(plural) for plural in after_stop]
 	temp = " ".join(after_stem)
 	clean_test.append(temp)

# TFIDF vector of the clean data(after removing punctuation , stemming and stop_words)
test_doc = bow_vec.transform(clean_test)
print test_doc.shape

test_doc = tfidf_vec.transform(test_doc)
print test_doc.shape


#svd.fit(vectors_test)
lsi_test = svd.transform(test_doc)
labels_test = twenty_test.target
print 'Dimensions of testing data are:',str(lsi_test.shape)

train_data = {'x' : lsi_train, 'y': labels_train}
test_data = {'x' : lsi_test, 'y': labels_test}

data = {'train': train_data, 'test': test_data}

with open('multiclass_db.pickle','wb') as f:
  pickle.dump(data,f)
