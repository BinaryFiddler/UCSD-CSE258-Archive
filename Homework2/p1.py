import numpy #maxtrix operation and linear algebra
import urllib #loading data from web
import csv
import scipy.optimize
import random

with open('winequality-white.csv', 'r') as f:
    reader = csv.reader(f, delimiter=';')
    wine = []
    for row in reader:
        wine = wine + [row]
f.close()

def dataProcessing(w):
    w = w[1:]
    for i in range(0, len(w)):
        for j in range(0, len(w[0])):
            w[i][j] = float(w[i][j])
    return w

print "Reading and processing data..."
wine = dataProcessing(wine)
train = wine[:len(wine)/3]
validation = wine[len(wine)/3+1: 2*len(wine)/3]
test = wine[2*len(wine)/3+1:]
print "done"

def feature(datum, ablation):
  feat = [1]
  for i in range(0, 11):
      if i == ablation:
          continue
      else:
          feat.append(datum[i])
  return feat


#compute MSE
def mean_squared_error(a, b):
    c = numpy.subtract(a, b)
    c = c ** 2
    return numpy.sum(c) / len(c)

def compute_MSE(i):
    #load X and y with training data
    X = [feature(d, i) for d in train]
    y = [d[11] for d in train]

    X = numpy.matrix(X)
    y = numpy.matrix(y)
    thetas = numpy.linalg.inv(X.T * X) * X.T * y.T

    #update X and y with test data
    X = [feature(d, i) for d in test]
    y = [d[11] for d in test]

    X = numpy.matrix(X)
    y = numpy.matrix(y)

    predicted = X * thetas
    predicted = predicted.T
    print mean_squared_error(predicted.A1, y.A1)
    return mean_squared_error(predicted.A1, y.A1)

MSE = [compute_MSE(i) for i in range(0, 11)]
