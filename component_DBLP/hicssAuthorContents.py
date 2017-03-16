# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:58:35 2017

@author: secoder
"""

import io
import nltk
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from wordcloudit import wordcloudit
import matplotlib.pyplot as plt

import numpy as np

"""
show contents in the same cluster
"""
def showcontents(labels, nth, allcontents):
    contents = []
    idx = np.where(labels == nth)
    idx = np.array(idx)
    idx = idx.flatten()
    for i in idx:
        contents.append(allcontents[i])
        
    return contents

def digstring(s):
    for i in s:
        if i.isdigit():
            return True
    return False

tokenizer = RegexpTokenizer(r'\w+')

#titledict={}
rawtitles = []
titles = []
newtitles = []
sw = set(nltk.corpus.stopwords.words('english'))


filename = './HICSS_titles.txt'


with io.open(filename,'r',encoding='utf8') as f:
    #text = f.read()
    for line in f:
        #hash_object = hashlib.md5(line.encode('utf-8'))
        #titledict[hash_object.hexdigest()]=newline
        #titledict[hash_object.hexdigest()] = [w for w in line.split() if w.lower() not in sw]
        
        rawtitles.append(line)
        newline=tokenizer.tokenize(line)
        # collect all the words except digtals and stopwords
        newline= ' '.join([w for w in newline if (w.lower() not in sw) & ~(digstring(w))])
        titles.append(newline)
        
        
filename = './HICSS_authors.txt'

authors = []
authorcontents = []

i = 0
with io.open(filename,'r',encoding='utf8') as f:
    for line in f:
        # split the authors by ','
        newline = line.split(",")
        # remove the last '\n' 
        newline.remove('\n')
        for name in newline:
            if name not in authors:
                authors.append(name)
                authorcontents.append(titles[i])
            else:    
                idx = authors.index(name)
                authorcontents[idx] = ' '.join([authorcontents[idx],titles[i]])
                        
        i = i + 1
        
vectorizer = CountVectorizer(max_df=0.95, min_df=1,stop_words='english')

X = vectorizer.fit_transform(authorcontents)

analyze = vectorizer.build_analyzer()



Xarray = X.toarray()
hist = sum(Xarray)

plt.plot(hist)

transformer = TfidfTransformer()

tfidf = transformer.fit_transform(Xarray)

featurenames = vectorizer.get_feature_names()

# do the clustering

# number of clusters
n = 10 

km=KMeans(n_clusters=n, init='k-means++',n_init=10, verbose=1)
km.fit(tfidf)

# show the word cloud of the first cluster
wordcloudit(' '.join(showcontents(km.labels_, 0, authorcontents)))
