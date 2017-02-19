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

print allRatings[0]
print userRatings[:5]



# globalAverage = sum(allRatings) / len(allRatings)
# userAverage = {}
# for u in userRatings:
#   userAverage[u] = sum(userRatings[u]) / len(userRatings[u])
#
# predictions = open("predictions_Rating.txt", 'w')
# for l in open("pairs_Rating.txt"):
#   if l.startswith("userID"):
#     #header
#     predictions.write(l)
#     continue
#   u,i = l.strip().split('-')
#   if u in userAverage:
#     predictions.write(u + '-' + i + ',' + str(userAverage[u]) + '\n')
#   else:
#     predictions.write(u + '-' + i + ',' + str(globalAverage) + '\n')
#
# predictions.close()
