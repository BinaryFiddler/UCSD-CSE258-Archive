import numpy
import urllib
import scipy.optimize
import random
from sklearn.decomposition import PCA
from collections import defaultdict
import matplotlib.pyplot as plt


### PCA on wine reviews ###

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
dataFile = open("winequality-white.csv")
header = dataFile.readline()
fields = ["constant"] + header.strip().replace('"','').split(';')
featureNames = fields[:-1]
labelName = fields[-1]
lines = [[] + [float(x) for x in l.split(';')] for l in dataFile]
X = [l[:-1] for l in lines]
y = [l[-1] > 5 for l in lines]
print "done"

X_train = X[:int(len(X)/3)]
y_train = y[:int(len(y)/3)]
X_validate = X[int(len(X)/3):int(2*len(X)/3)]
y_validate = y[int(len(y)/3):int(2*len(y)/3)]
X_test = X[int(2*len(X)/3):]
y_test = y[int(2*len(X)/3):]

train_mse = []
test_mse = []
dimension = []

def feature(datum, dimen):
  feat = [1]
  for i in range(0, dimen):
      feat.append(datum[i])
  return feat

def mean_squared_error(a, b):
    c = numpy.subtract(a, b)
    c = c ** 2
    return numpy.sum(c) / len(c)

for i in range(1, 12):
    #determine pca
    pca = PCA(n_components=i)
    pca.fit(X_train)

    #apply demensionality reduction
    X_train_reduced = pca.transform(X_train)
    X_test_reduced = pca.transform(X_test)

    #compute theta for regressor
    X = [feature(d, i) for d in X_train_reduced]
    y = [d for d in y_train]
    X = numpy.matrix(X)
    y = numpy.matrix(y)
    thetas = numpy.linalg.inv(X.T * X) * X.T * y.T

    #compute mse on training data
    predicted = X * thetas
    predicted = predicted.T
    dimension.append(i)
    train_mse.append(mean_squared_error(predicted.A1, y.A1))

    #compute mse on test data
    X = [feature(d, i) for d in X_test_reduced]
    y = [d for d in y_test]
    X = numpy.matrix(X)
    y = numpy.matrix(y)
    predicted = X * thetas
    predicted = predicted.T
    test_mse.append(mean_squared_error(predicted.A1, y.A1))

fig, ax = plt.subplots()
ax.plot(dimension, train_mse, 'r', label='train mse')
ax.plot(dimension, test_mse, 'g', label='test mse')
legend = ax.legend(loc='upper center', shadow=True)
plt.show()
