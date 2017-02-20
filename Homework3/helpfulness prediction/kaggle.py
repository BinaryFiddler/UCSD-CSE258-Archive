import gzip
import numpy as np
from collections import defaultdict

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

print "Reading data..."
data = list(readGz("train.json.gz"))
print "done"

train = data[:100000]
test = data[100000:]
x = []
y = []
for l in train:
    helpInfo, rate, num =l['helpful'],l['rating'], len(l['reviewText'].split(" "))
    out = helpInfo['outOf']
    nhelp = helpInfo['nHelpful']
    if out > 0:
        y.append(nhelp * 1.0 / out)
        ls = [1, num*1.0, rate*1.0]
        x.append(ls)

x = np.matrix(x)
y = np.matrix(y)
theta = np.linalg.inv(x.T * x) * x.T * y.T

map1 = {}
map2 = {}
for l in readGz("test_Helpful.json.gz"):
  user,item = l['reviewerID'],l['itemID']
  key = str(user) + '###' + str(item)
  rate, num = l['rating'], len(l['reviewText'].split(" "))
  map1[key] = rate
  map2[key] = num

predictions = open("predictions_Helpful.txt", 'w')
for l in open("pairs_Helpful.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i,outOf = l.strip().split('-')
  outOf = int(outOf)
  key = u + '###' + i
  rate = map1[key]
  num = map2[key]
  ls = [1, float(num), float(rate)]
  predict = np.dot(ls, theta)
  predictions.write(u + '-' + i + '-' + str(outOf) + ',' + str(outOf * float(predict)) + '\n')
