import numpy #maxtrix operation and linear algebra
import urllib #loading data from web
import scipy.optimize
import random

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
print "done"

def feature(datum):
  feat = [1]
  feat.append(datum['review/timeStruct']['year'])
  return feat

X = [feature(d) for d in data]
y = [d['review/overall'] for d in data]

X = numpy.matrix(X)
y = numpy.matrix(y)
thetas = numpy.linalg.inv(X.T * X) * X.T * y.T

predicted = X * thetas
predicted = predicted.T
def mean_squared_error(a, b):
    c = numpy.subtract(a, b)
    c = c ** 2
    return numpy.sum(c) / len(c)

#0.490043819858
print mean_squared_error(predicted.A1, y.A1)

#[ -3.91707489e+01,   2.14379786e-02]
print thetas
