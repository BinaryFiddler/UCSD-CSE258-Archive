import numpy #maxtrix operation and linear algebra
import urllib #loading data from web
import scipy.optimize
import random
import math

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
print "done"

def feature(datum):
  feat = [1]
  f = datum['review/timeStruct']['year']
  for i in range(1999, 2012):
      if f == i:
          feat.append(1)
      else:
          feat.append(0)
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

#0.48915189521
print mean_squared_error(predicted.A1, y.A1)

#[4.05154639  0.04845361  0.26663543  0.2247694  -0.29273378 -0.33610958 -0.29110335 -0.19133134 -0.24438704 -0.20546893 -0.16067206 -0.1301398 -0.15473656 -0.11142557]
print thetas.T
