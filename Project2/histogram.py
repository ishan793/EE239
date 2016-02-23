from sklearn.datasets import fetch_20newsgroups
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import text
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

cat_ct=['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware']
cat_ra=['rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']

'''ans=[]
all_cats = list(fetch_20newsgroups(subset='train').target_names)
for cat in all_cats:
    train_cat = fetch_20newsgroups(subset='train',categories=[cat],remove=('headers','footers','quotes'))
    ans.append(len(train_cat.data))

print ans    
'''
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
plt.show()