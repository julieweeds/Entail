__author__ = 'Julie'


import json, random,numpy,conf,sys


class EntailClassifier:
    cv=5  #number of cross-validation splits
    randomseed=42   #for cross-validation


    def __init__(self,pairfilename,freqfilename):

        self.pairfile=pairfilename
        self.freqfile=freqfilename
        self.pairmatrix = "" #will be numpy array which stores triple of w1,w1,1/0
        self.cv_idx= "" #will be index which indicates which split item in self.pairmatrix is in
        self.nopairs=0 #total number of pairs in data
        self.testing=False

        self.readpairs()
#        self.readtotals()

    def readpairs(self):

        print "Reading "+self.pairfile
        json_data = open(self.pairfile)
        data = json.load(json_data)
        json_data.close()
        rand_idx=[]

        self.validate_pairs(data) #checks datafile and sets self.nopairs

        for i in range(self.nopairs):
            rand_idx.append(i%EntailClassifier.cv)
            #print rand_idx

        random.seed(EntailClassifier.randomseed)
        random.shuffle(rand_idx)
        #print rand_idx

        self.pairmatrix =numpy.array(data)
        self.cv_idx=numpy.array(rand_idx)

        #refer to each split using matrix[idx==split]
        if self.testing:
            for split in range(EntailClassifier.cv):
                print str(len(self.pairmatrix[self.cv_idx==split]))

    def validate_pairs(self,data):

        self.nopairs=0
        lengthcheck=0
        ones=0
        zeros=0
        for item in data:
            self.nopairs+=1
            if len(item)==3:
                lengthcheck+=1
                if item[2]==1:
                    ones+=1
                elif item[2]==0:
                    zeros+=1

        #print self.nopairs,lengthcheck,ones,zeros
        if lengthcheck != self.nopairs:
            print "Warning Data error: number of pairs is "+str(self.nopairs)
            print "Number which have list length 3 is "+str(lengthcheck)
            exit(1)
        if ones+zeros != self.nopairs:
            print "Data error: number of pairs is "+str(self.nopairs)
            print "Number of ones is "+str(ones)
            print "Number of zeros is "+str(zeros)
            exit(1)
        print "Validated data:  "+str(self.nopairs)+" pairs"

    def test1(self,split,method,args):
        #method to test classification method in one split of data

        if method=="freq":
            threshold=float(args.pop())
            return self.testFreqThresh1(split,threshold)
        else:
            print "Error: Unknown method of classification "+method
            exit(1)

    def testFreqThresh1(self,split,threshold):
        #method to test frequency threshold in one cross-val split of data
        #split is the cv split to test in
        #threshold is the frequency threshold to test
        print threshold
        for [word1,word2,result] in self.pairmatrix[self.cv_idx==split]:
            print word1,word2,result

if __name__ == "__main__":
    parameters=conf.configure(sys.argv)
    myEntClassifier=EntailClassifier(parameters["pairfile"],parameters["freqfile"])
    #myEntClassifier.test1(1,"freq",[0])