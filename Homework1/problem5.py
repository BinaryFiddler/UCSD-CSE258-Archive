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

# Extract the features vector for each wine entry
def feature(datum):
  feat = []
  for i in range(0, 11):
    feat.append(datum[i])
  return feat

# Read the feature vector and the label data from training data
X_train = [feature(d) for d in train]
y_train = [d[11] > 5 for d in train]

X_test = [feature(d) for d in test]
y_test = [d[11] > 5 for d in test]

# print X_train


# Create a support vector classifier object, with regularization parameter C = 1000
clf = svm.SVC(C=1000)
clf.fit(X_train, y_train)

train_predictions = clf.predict(X_train)
test_predictions = clf.predict(X_test)


def count_acc(predict, original):
    count = 0.0
    for i in range(0, len(predict)):
        if predict[i] == original[i]:
            count = count + 1
    return count

# 1.0
# 0.668027766435
print count_acc(train_predictions.tolist(), y_train) / len(y_train)
print count_acc(test_predictions.tolist(), y_test) / len(y_test)
