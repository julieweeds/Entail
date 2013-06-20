__author__ = 'Julie'

from bitsbobs import untag
import math

class WordEntry:

    def __init__(self,fields):
        if len(fields)==3:
            (self.word,self.pos)=untag(fields[0])
            self.freq=fields[1]
            self.width=fields[2]
        elif len(fields)==2:
            (self.word,self.pos)=untag(fields[0])
            self.freq=fields[1]
            self.width=0
        else:
            print "Warning: invalid entry "+fields
        self.simdict={} #dictionary to store mapping from word to similarity score
        self.rankdict={} #dictionary to store mapping from word to rank in neighbour list
        self.paircount=0 #number of evaluation pairs involved in
        self.featdict={} #dict of features and scores
        self.precisiondict={} #to store values of WeedsPrecision
        self.min_precisiondict={} #to store values of ClarkeDE
        self.invCLdict={} #to store values of invCL (lenci)

    def addwordtodicts(self,word):
        self.simdict[word]=0
        self.rankdict[word]=float("inf")
    def addscorestodicts(self,word,sim,rank):
        if word in self.simdict.keys():
            self.simdict[word]=sim
            self.rankdict[word]=rank
            return 1
        else:
            return 0

    def writecache(self,outstream):
        if len(self.simdict.keys())>0:
            outstream.write(self.word)
            for w2 in self.simdict.keys():
                outstream.write("\t"+w2+"\t"+str(self.simdict[w2])+"\t"+str(self.rankdict[w2]))
            outstream.write("\n")

    def readfromcache(self,w2,sc,rank):
        self.simdict[w2]=sc
        self.rankdict[w2]=rank
        if sc>0:
            return 1
        else:
            return 0


    def addfeature(self,feat,score):
        #assumes only see a feature once per entry
        if feat!="___FILTERED___":
            self.featdict[feat]=score

    def precision(self,word):
        #calculate precision of retrieval of word's features by self
        #i.e., for each feature in self, is it in word?
        #additive model (token-based or mi based depending on vector file
        #for recall, call word.precision(self)
        if word in self.precisiondict.keys():
            return self.precisiondict[word]
        else:
            if len(self.featdict.keys())==0 or len(word.featdict.keys())==0:
                print "Warning: zero vectors"
                pre=0
            else:
                num=0
                den=0
                for feat in self.featdict.keys():
                    if feat in word.featdict.keys():
                        num+=self.featdict[feat]
                    den+=self.featdict[feat]
                pre = float(num)/float(den)
            self.precisiondict[word]=pre
            return pre

    def min_precision(self,word):
        #calculate precision of retrieval of word's features by self
        #i.e., for each feature in self, is it in word?
        #difference weighted model (token-based or mi based depending on vector file) as used by Clarke
        #for recall, call word.precision(self)
        if word in self.min_precisiondict.keys():
            return self.min_precisiondict[word]
        else:
            if len(self.featdict.keys())==0 or len(word.featdict.keys())==0:
                print "Warning: zero vectors"
                pre=0
            else:
                num=0
                den=0
                for feat in self.featdict.keys():
                    if feat in word.featdict.keys():
                        num+=min(self.featdict[feat],word.featdict[feat])
                    den+=self.featdict[feat]
                pre = float(num)/float(den)
            self.min_precisiondict[word]=pre
            return pre

    def invCL(self,word):
        if word in self.invCLdict.keys():
            return self.invCLdict[word]
        else:
            precision = self.min_precision(word)
            recall = word.min_precision(self)
            invCLval=math.pow(precision*(1-recall),0.5)
            self.invCLdict[word]=invCLval
            return invCLval



