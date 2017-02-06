import numpy
import urllib
import scipy.optimize
import random
from math import exp
from math import log
from sklearn.decomposition import PCA
import copy


random.seed(0)

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
dataFile = open("winequality-white.csv")
header = dataFile.readline()
fields = ["constant"] + header.strip().replace('"','').split(';')
featureNames = fields[:-1]
labelName = fields[-1]
lines = [[1.0] + [float(x) for x in l.split(';')] for l in dataFile]
X = [l[:-1] for l in lines]
y = [l[-1] > 5 for l in lines]
print "done"

X_train = X[:int(len(X)/3)]
X_reduce = copy.deepcopy(X_train)

for i in range(0, len(X_train[0])):
    sum = 0
    for j in range(0, len(X_train)):
        sum = sum + X_train[j][i]
    for j in range(0, len(X_train)):
        X_reduce[j][i] = sum / len(X_train)


def recon_error(X_Orig, X_Compressed):
    error = 0
    for i in range(0, len(X_Orig)):
        for j in range(0, len(X_Orig[0])):
            error = error + (X_Compressed[i][j] - X_Orig[i][j]) * (X_Compressed[i][j] - X_Orig[i][j])
    return error

print recon_error(X_train, X_reduce)
