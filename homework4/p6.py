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

tf1 = {}
for word in ''.join([c for c in data[0]['review/text'].lower() if not c in punctuation]).split():
  # only using 1000 features
  if word in words:
    if word in tf1:
      tf1[word] += 1
    else:
      tf1[word] = 1
    idf[word] = 0

tf2 = {}
for word in ''.join([c for c in data[1]['review/text'].lower() if not c in punctuation]).split():
  # only using 1000 features
  if word in words:
    if word in tf2:
      tf2[word] += 1
    else:
      tf2[word] = 1
    idf[word] = 0

for word in idf:
  for d in data:
    r = ''.join([c for c in d['review/text'].lower() if not c in punctuation])
    r = r.split()
    if word in r:
      idf[word] += 1

for key in idf:
  idf[key] = math.log10(5000 / float(idf[key]))

tf_idf1 = {}
for key in tf1:
  tf_idf1[key] = tf1[key] * idf[key] 

tf_idf2 = {}
for key in tf2:
  tf_idf2[key] = tf2[key] * idf[key] 

def find_cosine_sim(tf_idf1, tf_idf2):
  nominator = 0.0
  deno_part1 = 0.0
  deno_part2 = 0.0
  for key in tf_idf1:
    if key in tf_idf2:
      nominator += tf_idf1[key] * tf_idf2[key]
  for key in tf_idf1:
      deno_part1 += tf_idf1[key] ** 2 
  for key in tf_idf2:
      deno_part2 += tf_idf2[key] ** 2
  denominator = math.sqrt(deno_part1) * math.sqrt(deno_part2)
  return nominator / denominator

print find_cosine_sim(tf_idf1, tf_idf2)