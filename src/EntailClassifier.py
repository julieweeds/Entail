__author__ = 'Julie'


import json, random,numpy,conf,sys,re,math
from sep import Separator

wordposPATT=re.compile('(.*)/(.*)')

def untag(wordpos):

    matchobj=wordposPATT.match(wordpos)
    if matchobj:
        (word,pos)=(matchobj.group(1),matchobj.group(2))
        return (word,pos)
    else:
        print "Warning: Does not match word/pos pattern: "+wordpos
        return ("","")

def fscore(TP,FP,FN):
    if TP+FP==0:
        precision = 1
    else:
        precision = float(TP)/(float(TP)+float(FP))
    if TP+FN==0:
        recall = 1
    else:
        recall=float(TP)/(float(TP)+float(FN))
    f = 2*precision*recall/(precision+recall)
    return(precision,recall,f)

def mymean(list,k):
    n=0
    total=0
    totalsquare=0
    print list
    for item in list:
        n+=1
        total+=float(item)
        totalsquare+=float(item)*float(item)

    mean = total/n
    var = totalsquare/n - mean*mean
    if var<0:
        print "Warning: negative variance "+str(var)
        var=0
    sd = math.pow(var,0.5)
    int = k*sd/math.pow(len(list),0.5)
    return (mean,sd,int)


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


class EntailClassifier:
    cv=5  #number of cross-validation splits
    randomseed=42   #for cross-validation
    k=1.96 #for 95% confidence interval


    def __init__(self,pairfilename,freqfilename):

        self.pairfile=pairfilename
        self.freqfile=freqfilename
        self.pairmatrix = "" #will be numpy array which stores triple of w1,w1,1/0
        self.cv_idx= "" #will be index which indicates which split item in self.pairmatrix is in
        self.nopairs=0 #total number of pairs in data
        self.testing=False
        self.entrydict={} #to store freq info about words

        self.readpairs()
        self.readtotals()

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

    def readtotals(self):

        instream=open(self.freqfile,'r')
        print "Reading "+self.freqfile
        linesread=0
        for line in instream:
            fields=line.rstrip().split('\t')
            if len(fields) < 2:
                print "Discarding line "+line+" : "+str(len(fields))
            else:
                entry=WordEntry(fields)
                if entry.word == "":
                    print "Discarding line "+line
                else:
                    self.entrydict[entry.word]=entry
            linesread+=1
        print "Read "+str(linesread)+" lines"
        print "Size of WordEntry dict is "+str(len(self.entrydict))
        instream.close()

    def traintest(self,method):
        accuracy=[]
        for split in range(EntailClassifier.cv):
            threshold=self.train1(split,method)
            accuracy.append(self.test1(split,method,[threshold]))
        (mean,sd,int)=mymean(accuracy,EntailClassifier.k)
        print "Results for "+method+" are mean: "+str(mean)+" sd: "+str(sd)+" interval: +-"+str(int)



    def train1(self,split,method):
        #method to test classification method in one split of data

        if method=="freq":
            return self.trainFreqThresh1(split)
        if method=="zero_freq":
            return self.train0Freq1(split)
        else:
            print "Error: Unknown method of classification "+method
            exit(1)

    def trainFreqThresh1(self,split):

        print"Training split "+str(split)
        positives=[]
        negatives=[]
        done=0
        for [word1,word2,result] in self.pairmatrix[self.cv_idx!=split]:
            #if word1 in self.entrydict.keys():
            #    if word2 in self.entrydict.keys():
            diff = float(self.entrydict[word2].freq)-float(self.entrydict[word1].freq)
                #else:
                #    print "Error: no frequency information for "+word2
            #else:
                #print "Error: no frequency information for "+word1
            if int(result)==1:
                positives.append(diff)
            else:
                negatives.append(diff)
            done+=1
            if done%1000==0:
                print "Trained on "+str(done)
        print len(positives),len(negatives)

        threshold = Separator.separate(positives,negatives,trials=100000)
        return threshold


    def train0Freq1(self,split):
        #dummy to return freq threshold of 0
        return 0

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
        print "Testing threshold "+str(threshold)+" on split "+str(split)

        correct=0
        wrong=0
        total=0
        TP=0
        TN=0
        FP=0
        FN=0
        for [word1,word2,result] in self.pairmatrix[self.cv_idx==split]:
            #print word1,word2,result
            diff=threshold
            #if word1 in self.entrydict.keys():
            #    if word2 in self.entrydict.keys():
            diff = float(self.entrydict[word2].freq)-float(self.entrydict[word1].freq)


            #    else:
            #        print "Error: no frequency information for "+word2
            #else:
            #    print "Error: no frequency information for "+word1
            if diff>threshold:
                predict=1
            else:
                predict=0
            #print word1,word2,diff,predict,result
            if int(predict)==int(result):
                correct+=1
                if predict==1:
                    TP+=1
                else:
                    TN+=1
            else:
                wrong+=1
                if predict==1:
                    FP+=1
                else:
                    FN+=1
            total+=1
            #if total%10==0:
             #   break
        accuracy = float(correct)/float(total)
        (p,r,f)=fscore(TP,FP,FN)
        print "Correct: "+str(correct)+" Wrong: "+str(wrong)+" Total: "+str(total)+" Accuracy: "+str(accuracy)
        print "TP: "+str(TP)+" TN: "+str(TN)+" FP: "+str(FP)+" FN: "+str(FN)
        print "Precision: "+str(p)+" Recall: "+str(r)+" F: "+str(f)
        return accuracy


if __name__ == "__main__":
    parameters=conf.configure(sys.argv)
    myEntClassifier=EntailClassifier(parameters["pairfile"],parameters["freqfile"])
    #myEntClassifier.test1(0,"freq",[0])
    myEntClassifier.traintest("freq")