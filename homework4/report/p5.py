import numpy as np
import urllib
import scipy.optimize
import random
from collections import defaultdict
import nltk
import string
from nltk.stem.porter import *
from sklearn import linear_model
import math

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

### Just the first 5000 reviews

print "Reading data..."
data = list(parseData("beer_50000.json"))[:5000]
print "done"

idf = {}
for word in ['foam', 'smell', 'banana', 'lactic', 'tart']:
  idf[word] = 0

punctuation = set(string.punctuation)
for word in ['foam', 'smell', 'banana', 'lactic', 'tart']:
  for d in data:
    r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
    r = r.split()
    if word in r:
      idf[word] += 1

for key in idf:
  idf[key] = math.log10(len(data) / idf[key])
print idf

tf = {'foam':0, 'smell':0, 'banana':0, 'lactic':0, 'tart':0}
tf_idf = {}

for key in idf:
  d = data[0]
  r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
  r = r.split()
  for w in r:
    if key == w:
      tf[key] += 1
  tf_idf[key] = tf[key] * idf[key] 

print tf_idf