import gzip
from collections import defaultdict

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
userRatings = defaultdict(list)

for l in readGz("train.json.gz"):
    user,item = l['reviewerID'],l['itemID']
    allRatings.append(l['rating'])
    userRatings[user].append(l['rating'])

alpha = 1.0 * sum(allRatings[:100000]) / 100000

error = 0
for i in range(100000, 200000):
    error = error + (allRatings[i] - alpha) ** 2
print alpha
print error / 100000
