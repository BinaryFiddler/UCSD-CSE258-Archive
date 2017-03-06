import numpy as np
import urllib
import scipy.optimize
import random
from collections import defaultdict
import nltk
import string
from nltk.stem.porter import *
from sklearn import linear_model
import operator

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

### Just the first 5000 reviews

print "Reading data..."
data = list(parseData("beer_50000.json"))[:5000]
print "done"

### 1000 most common unigrams
wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  for w in r.split():
    wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

words = set([x[1] for x in counts[:1000]])

idf = {}
for w in words:
  idf[w] = 0

for word in idf:
  for d in data:
    r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
    r = r.split()
    if word in r:
      idf[word] += 1
### Sentiment analysis

wordId = dict(zip(words, range(len(words))))

def feature(datum):
  feat = [0] * len(words)
  tf = {}
  for word in ''.join([c for c in datum['review/text'].lower() if not c in punctuation]).split():
  # only using 1000 features
    if word in words:
      if word in tf:
        tf[word] += 1
      else:
        tf[word] = 1
  tf_idf = {}
  for key in tf:
    tf_idf[key] = tf[key] * idf[key] 
    feat[wordId[key]] = tf_idf[key]
  feat.append(1) #offset
  return feat

X = [feature(d) for d in data]
y = [d['review/overall'] for d in data]

#With regularization
clf = linear_model.Ridge(1.0, fit_intercept=False)
clf.fit(X, y)
theta = clf.coef_
# predictions = clf.predict(X)
# The mean squared error
print("Mean squared error: %f"
      % np.mean((clf.predict(X) - y) ** 2))