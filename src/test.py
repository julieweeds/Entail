__author__ = 'Julie'

import json
import random
import numpy

file="../data/wn-noun-dependencies.json"
myseed=42 #seed for random number generator
json_data = open(file)

data = json.load(json_data)



json_data.close()

count=0
lengthcheck=0
ones=0
zeros=0
for item in data:
    count+=1
    if len(item)==3:
        lengthcheck+=1
        if item[2]==1:
            ones+=1
        elif item[2]==0:
            zeros+=1

print count,lengthcheck,ones,zeros

cv=5
rand_idx=[]
for i in range(count):
    rand_idx.append(i%cv)
#print rand_idx

random.seed(myseed)
random.shuffle(rand_idx)
print rand_idx

matrix =numpy.array(data)
idx=numpy.array(rand_idx)

#refer to each split using matrix[idx==split]

for split in range(cv):
    print str(len(matrix[idx==split]))



#for item in data:
#    split = rand_idx.pop()
#    item.append(split)



