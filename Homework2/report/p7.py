import numpy
import urllib
import scipy.optimize
import random
from sklearn.decomposition import PCA
from collections import defaultdict

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

pca = PCA(n_components=4)
pca.fit(X_train)
X_new = pca.transform(X_train)

X_restored = pca.inverse_transform(X_new)
X_restored = X_restored.tolist()

def recon_error(X_Orig, X_Compressed):
    error = 0
    for i in range(0, len(X_Orig)):
        for j in range(0, len(X_Orig[0])):
            error = error + (X_Compressed[i][j] - X_Orig[i][j]) * (X_Compressed[i][j] - X_Orig[i][j])
    return error

print recon_error(X_train, X_restored)
