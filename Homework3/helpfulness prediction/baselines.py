import gzip
import numpy as np
from collections import defaultdict

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

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

print theta

x = []
yhelp = []
yout = []
for l in test:
    helpInfo, rate, num =l['helpful'],l['rating'], len(l['reviewText'].split(" "))
    out = helpInfo['outOf']
    nhelp = helpInfo['nHelpful']
    yhelp.append(nhelp * 1.0)
    yout.append(out * 1.0)
    ls = [1, num*1.0, rate*1.0]
    x.append(ls)

predicted = np.matrix(x) * theta
predicted = np.array(predicted.T)
yout = np.array(yout)
realPrediction = np.multiply(predicted, yout)

def mae(a, b):
	error = np.sum(np.absolute(np.subtract(a, b)))
	return error

print mae(realPrediction, np.array(yhelp)) / 100000

