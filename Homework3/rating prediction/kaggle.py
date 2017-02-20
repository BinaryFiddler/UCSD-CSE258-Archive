import gzip
import numpy as np
from collections import defaultdict

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

data = list(readGz("train.json.gz"))
train = data
### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
userRatings = defaultdict(list)
ratings = {}
userBoughtItem = defaultdict(list)
itemBoughtByUser = defaultdict(list)
betaU = {}
betaI = {}

print "Reading data..."
for l in train:
    user,item,rating = l['reviewerID'],l['itemID'], l['rating']
    key = str(user) + "###" + str(item)
    ratings[key] = rating
    allRatings.append(rating)
    userBoughtItem[user].append(item)
    itemBoughtByUser[item].append(user)
    betaU[user] = 0
    betaI[item] = 0
print "done"

alpha = 1.0 * sum(allRatings[:100000]) / 100000
alpha = 0

iteration = 20
lamda = 7

for i in range(0, iteration):
	# print alpha, betaU['U989129959'], betaI['I734011860']
 	# update alpha
	alpha = (sum(ratings.values()) - sum(betaU.values()) - sum(betaI.values()))/ len(train)
	# update betaU
	for key in betaU:
		relavantItems = userBoughtItem[key]
		betaU[key] =  sum((ratings[str(key) + "###" + str(e)] - (alpha + betaI[e])) for e in relavantItems) / (lamda + len(relavantItems))
	# update betaI
	for key in betaI:
		relavantUsers = itemBoughtByUser[key]
		betaI[key] = sum((ratings[str(e) + "###" + str(key)] - (alpha + betaU[e])) for e in relavantUsers) / (lamda + len(relavantUsers)) 


predictions = open("predictions_Rating.txt", 'w')
for l in open("pairs_Rating.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i = l.strip().split('-')
  key = u + '###' + i
  bu = 0
  bi = 0
  if u in betaU:
    bu = betaU[u]
  if i in betaI:
    bi = betaI[i]
  predict = alpha + bu + bi
  if key in ratings:
    predict = ratings[key]
  predictions.write(u + '-' + i  + ',' + str(predict) + '\n')
