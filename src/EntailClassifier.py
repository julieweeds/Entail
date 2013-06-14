__author__ = 'Julie'


import json, random,numpy,conf,sys,re,math
from sep import Separator
import time,datetime

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
    #print list
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
        self.simdict={} #dictionary to store mapping from word to similarity score
        self.rankdict={} #dictionary to store mapping from word to rank in neighbour list

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


class EntailClassifier:
    cv=5  #number of cross-validation splits
    randomseed=42   #for cross-validation
    k=1.96 #for 95% confidence interval


    def __init__(self,pairfilename,freqfilename):

        self.pairfile=pairfilename
        self.freqfile=freqfilename
        self.simsfile=""
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

    def loadsims(self,simsfile,use_cache=False,make_cache=True):
        #is there a relevant cache of relevant sims? If so, load
        #otherwise first need to establish which word pairs we need to store similarities for using the pairmatrix
        #then read the simsfile and store the similarities
        #and write to cache
        self.simsfile=simsfile
        if use_cache:
            self.loadcachedsims()
        else:
            for [w1,w2,_r] in self.pairmatrix:
                #for each word (in each word pair) want to put the other word in its dictionary so a similarity will be stored if found in simsfile
                self.entrydict[w1].addwordtodicts(w2)
                self.entrydict[w2].addwordtodicts(w1)

            simstream=open(simsfile,'r')
            print "Reading "+simsfile
            linesread=0
            added=0
            ignored=0
            for line in simstream:
                linesread+=1
                line.rstrip()
                fields=line.split('\t')
                fields.reverse()
                (w1,_)=untag(fields.pop())
                if len(self.entrydict[w1].simdict)>0:
                    #don't care about sims for words not in evaluation
                    rank=1
                    while len(fields)>0:
                        (w2,_)=untag(fields.pop())
                        score=float(fields.pop())
                        added+=self.entrydict[w1].addscorestodicts(w2,score,rank) #will only add if pair is initialised
                        rank+=1
                else:
                    #print "Ignoring line "+str(linesread)+": "+w1
                    ignored+=1

                if linesread%100==0:
                    print "Read "+str(linesread)+" lines and ignored "+str(ignored)+" lines and stored "+str(added)+" similarities"
                    #break
            print "Read "+str(linesread)+" lines and ignored "+str(ignored)+" lines and stored "+str(added)+" similarities"
            simstream.close()
            if make_cache:
                self.makesimcache()

    def loadcachedsims(self):
        cachename=self.simsfile+".cached"
        instream=open(cachename,'r')
        print "Reading "+cachename
        linesread=0
        added=0
        for line in instream:
            linesread+=1
            line.rstrip()
            fields=line.split('\t')
            fields.reverse()
            #if linesread>1393: print fields
            w1=fields.pop()

            while len(fields)>0:
                w2=fields.pop()
                sc=float(fields.pop())
                rank=float(fields.pop())
                added+=self.entrydict[w1].readfromcache(w2,sc,rank)

            if linesread%100==0:
                print "Read "+str(linesread)+" lines and added "+str(added)+" similarities"
                #break
        print "Read "+str(linesread)+" lines and added "+str(added)+" similarities"
        instream.close()
        return

    def makesimcache(self):
        cachename=self.simsfile+".cached"
        outstream=open(cachename,'w')
        for w1 in self.entrydict.keys():
            self.entrydict[w1].writecache(outstream)

        outstream.close()
        return

    def traintest(self,method):
        scores={}

        scores["accuracy"]=[]
        scores["precision"]=[]
        scores["recall"]=[]
        scores["f1score"]=[]

        for split in range(EntailClassifier.cv):
            parameters=self.train1(split,method)
            (acc,pre,rec,f)=self.test1(split,method,parameters)
            scores["accuracy"].append(acc)
            scores["precision"].append(pre)
            scores["recall"].append(rec)
            scores["f1score"].append(f)

        for score in scores.keys():
            (mean,sd,int)=mymean(scores[score],EntailClassifier.k)
            print "Results for "+method+" are "+score+": mean= "+str(mean)+" sd= "+str(sd)+" interval= +-"+str(int)
            if score=="precision":
                premean=mean
            if score=="recall":
                recmean=mean

        f=2*premean*recmean/(premean+recmean)
        print "F1 of average precision and average recall is "+str(f)

    def train1(self,split,method):
        #method to test classification method in one split of data

        if method=="freq":
            return self.trainFreqThresh1(split)
        elif method=="zero_freq":
            return self.train0Freq1(split)
        elif method=="lin_freq":
            return self.trainlinfreq(split)
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
            if done%100==0:
                print "Trained on "+str(done)
        print len(positives),len(negatives)

        threshold = Separator.separate(positives,negatives,trials=1000000)
        return [threshold]


    def train0Freq1(self,split):
        #dummy to return freq threshold of 0
        return [0]

    def trainlinfreq(self,split):
        return [100,float("-inf")]

    def test1(self,split,method,args):
        #method to test classification method in one split of data
        # method
        if method=="freq" or method=="zero_freq":
            threshold=float(args.pop())
            #print threshold
            return self.testFreqThresh1(split,threshold)
        elif method=="lin_freq" or method=="lin_freq_thresh":
            (freqthresh,kthresh)=(float(args.pop()),float(args.pop()))
            #print "Thresholds ",kthresh,freqthresh
            return self.testlinfreq(split,kthresh,freqthresh)

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
        return (accuracy,p,r,f)


    def testlinfreq(self,split,kthresh,freqthresh):
        #method to test frequency threshold in one cross-val split of data
        #split is the cv split to test in
        #threshold is the frequency threshold to test
        print "Testing thresholds k= "+str(kthresh)+" and freq2-freq1 > "+str(freqthresh)+" on split "+str(split)

        correct=0
        wrong=0
        total=0
        TP=0
        TN=0
        FP=0
        FN=0
        for [word1,word2,result] in self.pairmatrix[self.cv_idx==split]:
            #print word1,word2,result
            diff=freqthresh
            #if word1 in self.entrydict.keys():
            #    if word2 in self.entrydict.keys():

            rank1=self.entrydict[word1].rankdict[word2]
            rank2=self.entrydict[word2].rankdict[word1]
            diff = float(self.entrydict[word2].freq)-float(self.entrydict[word1].freq)


            #    else:
            #        print "Error: no frequency information for "+word2
            #else:
            #    print "Error: no frequency information for "+word1
            if rank1 <= kthresh or rank2 <= kthresh:
                if diff>freqthresh:
                    predict=1
                else:
                    predict=0
                    #print word1,word2,diff,predict,result
            else:
                predict =0
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
        return (accuracy,p,r,f)


if __name__ == "__main__":
    parameters=conf.configure(sys.argv)
    starttime=time.time()
    print "Started at "+datetime.datetime.fromtimestamp(starttime).strftime('%Y-%m-%d %H:%M:%S')
    #print parameters

    myEntClassifier=EntailClassifier(parameters["pairfile"],parameters["freqfile"])
    #myEntClassifier.test1(0,"freq",[0])

    if "lin_freq" in parameters["methods"]:
        #need to load up thesaurus similarity file
        #probably want to cache the relevant similarity scores
        myEntClassifier.loadsims(parameters["simsfile"],parameters["use_cache"])
        #print myEntClassifier.entrydict["ambulance"].simdict
        #print myEntClassifier.entrydict["ambulance"].rankdict


    for method in parameters["methods"]:
        myEntClassifier.traintest(method)


    endtime=time.time()
    print "Finished at "+datetime.datetime.fromtimestamp(endtime).strftime('%Y-%m-%d %H:%M:%S')
    elapsed = endtime-starttime
    hourselapsed=int(elapsed/3600)
    minselapsed=int((elapsed-hourselapsed*3600)/60)
    print "Time taken is "+str(hourselapsed)+" hours and "+str(minselapsed)+" minutes."
