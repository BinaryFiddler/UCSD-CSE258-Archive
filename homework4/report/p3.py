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

bigramCount = {}
### Count bigrams
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  r = r.split()
  for i in range(0, len(r) - 1):
    pair = r[i] + " " + r[i+1]
    if pair in bigramCount:
      bigramCount[pair] = bigramCount[pair] + 1
    else:
      bigramCount[pair] = 1

unigramCount = {}
for d in data:
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  for w in r.split():
    if w in unigramCount:
      unigramCount[w] += 1
    else:
      unigramCount[w] = 1


combine = dict(bigramCount.items() + unigramCount.items())
combine_sort = sorted(combine.items(), key = operator.itemgetter(1), reverse = True)

words = []
for s in combine_sort[:1000]:
  words.extend([s[0]])

### Sentiment analysis

wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

def feature(datum):
  feat = [0]*len(words)
  r = ''.join([c for c in datum['review/text'].lower() if not c in punctuation])
  r = r.split()
  for w in r:
    if w in words:
      feat[wordId[w]] += 1
  for i in range(0, len(r) - 1):
    w = r[i] + " " + r[i+1]
    if w in words:
      feat[wordId[w]] += 1
  feat.append(1) #offset
  return feat

X = [feature(d) for d in data]
y = [d['review/overall'] for d in data]

#No regularization
#theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

#With regularization
clf = linear_model.Ridge(1.0, fit_intercept=False)
clf.fit(X, y)
theta = clf.coef_
predictions = clf.predict(X)
print("Mean squared error: %f"
      % np.mean((clf.predict(X) - y) ** 2))

weights = {}
for i in range(len(words)):
  for key, value in wordId.iteritems():
    if value == i:
      weights[key] = theta[i]
      break

weight_sort = sorted(weights.items(), key = operator.itemgetter(1), reverse = True)
print weight_sort[:5]
print weight_sort[-5:]

  



