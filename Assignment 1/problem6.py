import numpy
import urllib
import scipy.optimize
import random
import csv
from math import exp
from math import log

with open('winequality-white.csv', 'r') as f:
    reader = csv.reader(f, delimiter=';')
    wine = []
    for row in reader:
        wine = wine + [row]
f.close()

# Read data from CSV file
def dataProcessing(w):
    w = w[1:]
    for i in range(0, len(w)):
        for j in range(0, len(w[0])):
            w[i][j] = float(w[i][j])
    return w

# Split data into test data and training data
print "Reading and processing data..."
wine = dataProcessing(wine)
print "done"

def feature(datum):
  feat = []
  for i in range(0, 11):
    feat.append(datum[i])
  return feat

def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
  return 1.0 / (1 + exp(-x))

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    loglikelihood -= log(1 + exp(-logit))
    if not y[i]:
      loglikelihood -= logit
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  print "ll =", loglikelihood
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
  dl = [0.0]*len(theta)
  for i in range(len(X)):
    # Fill in code for the derivative
    logit = inner(X[i], theta)
    M =  [x * numpy.exp(-logit) / (1 + numpy.exp(-logit)) for x in X[i]]
    dl = [d + m for (d, m) in zip(dl, M)]
    # dl = dl + X[i] * numpy.exp(-logit) / (1 + numpy.exp(-logit))
    if not y[i]:
        dl = [d - x for (d, x) in zip(dl, X[i])]
        # dl = dl - X[i]
  R = [t * 2 * lam for t in theta]
  dl = [d - r for (d, r) in zip(dl, R)]
  # Negate the return value since we're doing gradient *ascent*
  return numpy.array([-x for x in dl])


X = [feature(d) for d in wine]
# Extract features and labels from the data
y = [d[11] > 5 for d in wine]

y_train = y[:len(y)/2]
y_test = y[len(y)/2:]

X_train = X[:len(X)/2]
X_test = X[len(X)/2:]

# If we wanted to split with a validation set:
#X_valid = X[len(X)/2:3*len(X)/4]
#X_test = X[3*len(X)/4:]

# Use a library function to run gradient descent (or you can implement yourself!)
theta,l,info = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X[0]), fprime, args = (X_train, y_train, 1.0))
print "Final log likelihood =", -l
# -1388.69674843

theta = numpy.matrix(theta)
X_test = numpy.matrix(X_test)

y_predict = X_test * theta.T
y_predict = [y > 0.0 for y in y_predict]

def count_acc(predict, original):
    count = 0.0
    for i in range(0, len(predict)):
        if predict[i] == original[i]:
            count = count + 1
    return count

# print count_acc(y_predict, y_test) / len(y_test)
print "Accuracy = ", count_acc(y_predict, y_test) / len(y_test)
# Compute the accuracy 0.76929358922
