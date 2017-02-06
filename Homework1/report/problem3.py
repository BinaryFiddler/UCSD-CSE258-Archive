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
train = wine[:len(wine)/2]
test = wine[len(wine)/2:]
print "done"

def feature(datum):
  feat = [1]
  for i in range(0, 11):
    feat.append(datum[i])
  return feat

#compute MSE
def mean_squared_error(a, b):
    c = numpy.subtract(a, b)
    c = c ** 2
    return numpy.sum(c) / len(c)

#load X and y with training data
X = [feature(d) for d in train]
y = [d[11] for d in train]

X = numpy.matrix(X)
y = numpy.matrix(y)
thetas = numpy.linalg.inv(X.T * X) * X.T * y.T

predicted = X * thetas
predicted = predicted.T

print mean_squared_error(predicted.A1, y.A1)

#update X and y with test data
X = [feature(d) for d in test]
y = [d[11] for d in test]

X = numpy.matrix(X)
y = numpy.matrix(y)

predicted = X * thetas
predicted = predicted.T



# 0.562457127767
print mean_squared_error(predicted.A1, y.A1)

# [[  2.56420278e+02   1.35421303e-01  -1.72994866e+00   1.02651152e-01
#     1.09038568e-01  -2.76775152e-01   6.34332169e-03   3.85023935e-05
#    -2.58652808e+02   1.19540565e+00   8.33006284e-01   9.79304364e-02]]
# print thetas.T
