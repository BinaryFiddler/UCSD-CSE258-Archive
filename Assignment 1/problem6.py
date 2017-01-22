import numpy
import urllib
import scipy.optimize
import random
import csv
from sklearn import svm

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
train = wine[:len(wine)/2]
test = wine[len(wine)/2:]
print "done"
