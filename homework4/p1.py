import numpy
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

print len(bigramCount)
sorted_bigram = sorted(bigramCount.items(), key=operator.itemgetter(1), reverse = True)
print sorted_bigram[:5]