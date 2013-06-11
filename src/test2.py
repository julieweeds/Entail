__author__ = 'juliewe'

import EntailClassifier
import bisect

wordpos="garment/N"

(word,pos)=EntailClassifier.untag(wordpos)
print word+" : "+pos


sortedlist=[2,3,4,10,15,20]
key = 15
print bisect.bisect_left(sortedlist,key)
print len(sortedlist) - bisect.bisect_right(sortedlist,key)
