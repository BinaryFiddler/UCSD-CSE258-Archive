import gzip
from collections import defaultdict

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
iteration = 15
lam = 0.2
alpha
beta1
beta2
userRatings = defaultdict(list)
for i in range(iteration):
    print i
    for l in readGz("train.json.gz"):
        alpha = 

      user,item = l['reviewerID'],l['itemID']
      helpful = l['helpful']
      nhelpful.append(helpful['nHelpful'])
      outof.append(helpful['outOf'])
      allRatings.append(l['rating'])
      userRatings[user].append(l['rating'])
