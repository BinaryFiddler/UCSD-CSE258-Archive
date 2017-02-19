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

#
# allRatings = []
# allLength = []
# nhelpful = []
# outof = []
# userRatings = defaultdict(list)
#
# for l in readGz("train.json.gz"):
#       user,item = l['reviewerID'],l['itemID']
#       allRatings.append(l['rating'])
#       allLength.append(len(l['reviewText']))
#       nhelpful.append(l['helpful']['nHelpful'])
#       outof.append(l['helpful']['outOf'])
#       userRatings[user].append(l['rating'])

def feature(datum):
  feat = [1]
  feat.append(len(datum['reviewText']))
  feat.append(datum['rating'])
  return feat

# building the training set
Xtrain = [feature(d) for d in data[:100000] if d['helpful']['outOf'] != 0]
ytrain = [d['helpful']['nHelpful'] / d['helpful']['outOf'] for d in data[:100000] if d['helpful']['outOf'] != 0]

print len(Xtrain), len(ytrain)

Xtrain = np.matrix(Xtrain)
ytrain = np.matrix(ytrain)
thetas = np.linalg.inv(Xtrain.T * Xtrain) * Xtrain.T * ytrain.T

# building the validation set
Xvalid = [feature(d) for d in data[100000:]]
yvalid = np.array([[d['helpful']['nHelpful'] for d in data[100000:]]])

predicted = np.matrix(Xvalid) * thetas
predicted = predicted.T
predicted = np.array(predicted)
yvalidOutOf = np.array([[d['helpful']['outOf'] for d in data[100000:]]])

def mae(a, b):
	error = np.sum(np.absolute(np.subtract(a, b)))
	return error


print yvalidOutOf
print yvalidOutOf.shape, type(yvalidOutOf)
print predicted.shape, type(predicted)

nHelpPrediction = np.multiply(predicted,yvalidOutOf)
print nHelpPrediction
print mae(nHelpPrediction, yvalid) / 100000
