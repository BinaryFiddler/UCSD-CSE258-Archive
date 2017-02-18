import gzip
from collections import defaultdict

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
nhelpful = [];
outof = [];
userRatings = defaultdict(list)
for l in readGz("train.json.gz"):
  user,item = l['reviewerID'],l['itemID']
  helpful = l['helpful']
  nhelpful.append(helpful['nHelpful'])
  outof.append(helpful['outOf'])
  allRatings.append(l['rating'])
  userRatings[user].append(l['rating'])

alpha = 1.0 * sum(nhelpful[:100000]) / sum(outof[:100000])
print "alpha: ", alpha

totalError = 0
for i in range(100000, 200000):
    if(outof[i] == 0):
        continue
    else:
        totalError = totalError + abs(1.0 * nhelpful[i] / outof[i] - alpha)

print "Mean absolute error:", totalError / len(outof) * 2
