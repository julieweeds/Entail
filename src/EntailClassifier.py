__author__ = 'Julie'


import json, random,numpy,conf,sys
from sep import Separator
import time,datetime
from wordEntry import WordEntry
from bitsbobs import untag, mymean,fscore, f_analyse
#from sklearn import svm

class EntailClassifier:
    cv=5  #number of cross-validation splits
    randomseed=42   #for cross-validation
    k=1.96 #for 95% confidence interval
    kneighs=1000
    do_ratio = True #whether to do ratio or difference in threshold prediction (CR or frequency)


    def __init__(self,pairfilename,freqfilename,use_cache=False):

        self.pairfile=pairfilename
        self.freqfile=freqfilename
        self.use_cache=use_cache
        self.paircache=False #use cache of pairs
        self.transformpairs=False #run function to remove negatives and reverse positives
        self.simsfile=""
        self.pairmatrix = "" #will be numpy array which stores triple of w1,w1,1/0
        self.cv_idx= "" #will be index which indicates which split item in self.pairmatrix is in
        self.nopairs=0 #total number of pairs in data
        self.testing=False
        self.verbose=False
        self.entrydict={} #to store freq info about words
        if EntailClassifier.do_ratio:
            self.comparestring="/"
        else:
            self.comparestring="-"


        self.readpairs()
        self.readtotals()

    def readpairs(self):

        if self.paircache:
            self.pairfile=self.pairfile+'.cached'
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
                if int(item[2])==1:
                    ones+=1
                elif int(item[2])==0:
                    zeros+=1
        #print data[0]
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
        print "Number of ones is "+str(ones)
        print "Number of zeros is "+str(zeros)

    def filter_pairs(self):
        #remove any pairs for which we don't have frequency/corpus information
        incount=0
        totalcount=0
        outcount=0
        todelete=[]
        print "Checking pairmatrix against frequency information"
        for i in range(len(self.pairmatrix)):
            [word1,word2,_]=self.pairmatrix[i]
            #print (word1,word2)
            #print self.entrydict.keys()
            if word1 in self.entrydict.keys() and word2 in self.entrydict.keys():
                incount+=1
            else:
                outcount+=1
                #save index in todelete list
                todelete.append(i)
            totalcount+=1
            if totalcount%1000==0:
                print "Checked "+str(totalcount)+" in = "+str(incount)+" out = "+str(outcount)


        print "Checked frequency information for "+str(totalcount)+" pairs: kept "+str(incount)+" and discarding "+str(outcount)+" ("+str(float(outcount)/float(totalcount))+")"
        self.pairmatrix=numpy.delete(self.pairmatrix,todelete,0)
        self.nopairs=len(self.pairmatrix)
        print "Size of pairmatrix is now "+str(self.nopairs)
        #print self.pairmatrix[0]

        print "Writing cache "+self.pairfile+".cached"
        with open(self.pairfile+".cached",'w') as outfile:
            json.dump(self.pairmatrix.tolist(),outfile)

    def transform_pairs(self):
        #transform dataset to be positives and reversed positives and save to file

        incount=0
        outcount=0
        totalcount=0
        todelete=[]
        toadd=[]
        newarray=[]
        print "Transforming dataset into positives and reversed positives"
        for i in range(len(self.pairmatrix)):
            [word1,word2,score]=self.pairmatrix[i]
            #print word1,word2,score
            if int(score) == 0:
                outcount+=1
                todelete.append(i) #list of indices to delete: current zeroes

            else:
                incount+=1
                toadd.append([word2,word1,0]) #list of new pairs: reversed ones
                newarray.append([word1,word2,1])
                newarray.append([word2,word1,0])
            totalcount+=1
            if totalcount%1000==0:
                print "Checked "+str(totalcount)+" pairs: in = "+str(incount)+" out = "+str(outcount)
        print "Checked "+str(totalcount)+" pairs: in = "+str(incount)+" out = "+str(outcount)
        self.pairmatrix=numpy.delete(self.pairmatrix,todelete,0) #delete zeroes
        newrows=numpy.array(toadd)
        self.pairmatrix = numpy.vstack((self.pairmatrix,newrows))
        self.nopairs=len(self.pairmatrix)
        print "Size of pairmatrix is now "+str(self.nopairs)
        print "Writing file "+self.pairfile+".transform"
        with open(self.pairfile+".transform",'w') as outfile:
            #json.dump(self.pairmatrix.tolist(),outfile)
            json.dump(newarray,outfile)


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
        if not self.paircache:
            self.filter_pairs()
        if self.transformpairs:
            self.transform_pairs()

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
                #print w1
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

                if self.verbose and linesread%100==0:
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

            if self.verbose and linesread%100==0:
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

    def loadvectors(self,vectorfile,use_cache=False,make_cache=True):
        self.vectorfile=vectorfile
        if use_cache:
            #self.loadvectorcache()
            self.vectorfile=self.vectorfile+".cached"
            make_cache=False

        for[w1,w2,_r] in self.pairmatrix:
            self.entrydict[w1].paircount+=1
            self.entrydict[w2].paircount+=1

        instream=open(self.vectorfile,'r')
        print "Reading "+self.vectorfile
        linesread=0
        lineswritten=0
        if make_cache:
            outstream=open(self.vectorfile+".cached",'w')
            print "Writing "+self.vectorfile+".cached"
        for line in instream:
            linesread+=1
            line=line.rstrip()
            fields=line.split('\t')
            fields.reverse()
            (w1,_)=untag(fields.pop())
            if w1 != "" and self.entrydict[w1].paircount>0:
                #store this vector
                if make_cache:
                    outstream.write(line)
                    lineswritten+=1
                while len(fields)>0:
                    w2=fields.pop()
                    sc=float(fields.pop())
                    self.entrydict[w1].addfeature(w2,sc)
            if self.verbose and linesread%1000==0:
                print "Read "+str(linesread)+" lines and written "+str(lineswritten)+" lines"

        if make_cache:
            outstream.close()
            print "Written "+str(lineswritten)+" lines"
        instream.close()
        print "Read "+str(linesread)+" lines"


    def traintest(self,method):
        scores={}

        scores["accuracy"]=[]
        scores["precision"]=[]
        scores["recall"]=[]
        scores["f1score"]=[]

        for split in range(EntailClassifier.cv):
            if method=="CR_svm":
                (acc,pre,rec,f)=self.svm_train(split,"CR")
            else:
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
        elif method=="lin":
            return self.trainlin(split)
        elif method=="CR" or method=="clarke":
            return self.trainCR(split)
        elif method=="CR_thresh" or method=="clarke_thresh" or method=="invCL":
            return self.trainCRThresh(split,method)
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
            if EntailClassifier.do_ratio:
                diff = float(self.entrydict[word2].freq)/float(self.entrydict[word1].freq)
            else:
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
            if self.verbose and done%1000==0:
                print "Trained on "+str(done)
        print len(positives),len(negatives)

        threshold = Separator.separate(positives,negatives,trials=1000000)
        return [threshold]

    def trainCR(self,split):
        if EntailClassifier.do_ratio:
            return[1]
        else:
            return [0]

    def train0Freq1(self,split):
        #dummy to return freq threshold of 0
        if EntailClassifier.do_ratio:
            return[1]
        else:
            return [0]

    def trainCRThresh(self,split,method):

        print"Training split "+str(split)
        positives=[]
        negatives=[]
        done=0
        for [word1,word2,result] in self.pairmatrix[self.cv_idx!=split]:

            #diff = float(self.entrydict[word2].freq)-float(self.entrydict[word1].freq)
            if method=="CR_thresh":
                precision = float(self.entrydict[word1].precision(self.entrydict[word2]))
                recall=float(self.entrydict[word2].precision(self.entrydict[word1]))
            elif method=="clarke_thresh":
                precision = float(self.entrydict[word1].min_precision(self.entrydict[word2]))
                recall=float(self.entrydict[word2].min_precision(self.entrydict[word1]))
            elif method=="invCL":
                precision = float(self.entrydict[word1].invCL(self.entrydict[word2]))
                recall=1 #invCL uses 1-recall anyway so threshold just on precision value
            else:
                print "Unknown CR method "+method
                exit(1)
            if recall == 0:
                ratio=0
                #hm=0
            else:
                if EntailClassifier.do_ratio:
                    ratio=precision/recall
                else:
                    ratio=precision-recall
                #hm=2*precision*recall/(precision+recall)
            if int(result)==1:
                positives.append(ratio)
            else:
                negatives.append(ratio)
            done+=1
            if self.verbose and done%1000==0:
                print "Trained on "+str(done)
        print len(positives),len(negatives)

        threshold = Separator.separate(positives,negatives,trials=1000000,integer=False)
        return [threshold]

    def trainlinfreq(self,split):
        if EntailClassifier.do_ratio:
            return [EntailClassifier.kneighs,1]
        else:
        #    return [EntailClassifier.kneighs,float("-inf")]
            return [EntailClassifier.kneighs,0]

    def trainlin(self,split):
        return [EntailClassifier.kneighs,0]

    def test1(self,split,method,args):
        #method to test classification method in one split of data
        # method
        if method=="lin_freq" or method =="lin":
            (freqthresh,kthresh)=(args[1],args[0])

            print "Testing thresholds k= "+str(kthresh)+" and freq2"+self.comparestring+"freq1 > "+str(freqthresh)+" on split "+str(split)
        else:
            [threshold]=args
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
            #if word1 in self.entrydict.keys():
            #    if word2 in self.entrydict.keys():
            predict = self.predict(method,word1,word2,args)

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
            if self.verbose and total%100==0:
                print "Completed "+str(total)+" tests"

        accuracy = float(correct)/float(total)
        (p,r,f)=fscore(TP,FP,FN)
        print "Correct: "+str(correct)+" Wrong: "+str(wrong)+" Total: "+str(total)+" Accuracy: "+str(accuracy)
        print "TP: "+str(TP)+" TN: "+str(TN)+" FP: "+str(FP)+" FN: "+str(FN)
        print "Precision: "+str(p)+" Recall: "+str(r)+" F: "+str(f)
        return (accuracy,p,r,f)

    def predict(self,method,word1,word2,args):
        if method=="freq" or method=="zero_freq":
            return self.freqpredict(word1,word2,args)
        elif method=="lin_freq" or method=="lin" or method=="lin_freq_thresh":
            return self.linpredict(word1,word2,args)
            #print "Thresholds ",kthresh,freqthresh
        elif method in ["CR","CR_thresh","clarke","clarke_thresh","invCL"]:
            return self.CRpredict(word1,word2,args,method)
        else:
            print "Error: Unknown method of classification "+method
            exit(1)

    def linpredict(self,word1,word2,args):
        (freqthresh,kthresh)=(args[1],args[0])
        rank1=self.entrydict[word1].rankdict[word2]
        rank2=self.entrydict[word2].rankdict[word1]
        if EntailClassifier.do_ratio:
            diff = float(self.entrydict[word2].freq)/float(self.entrydict[word1].freq)

        else:
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
        return predict

    def freqpredict(self,word1,word2,args):
        [threshold]=args
        if EntailClassifier.do_ratio:
            diff = float(self.entrydict[word2].freq)/float(self.entrydict[word1].freq)
        else:
            diff = float(self.entrydict[word2].freq)-float(self.entrydict[word1].freq)


        #    else:
        #        print "Error: no frequency information for "+word2
        #else:
        #    print "Error: no frequency information for "+word1
        if diff>threshold:
            predict=1
        else:
            predict=0
        return predict

    def CRpredict(self,word1,word2,args,method):
        [threshold]=args

        if method=="CR" or method=="CR_thresh":
            precision = float(self.entrydict[word1].precision(self.entrydict[word2]))
            recall=float(self.entrydict[word2].precision(self.entrydict[word1]))
        elif method=="clarke" or method=="clarke_thresh":
            precision = float(self.entrydict[word1].min_precision(self.entrydict[word2]))
            recall=float(self.entrydict[word2].min_precision(self.entrydict[word1]))
        elif method=="invCL":
            precision = float(self.entrydict[word1].invCL(self.entrydict[word2]))
            recall=1
        else:
            print "Unknown CR method "+method
            exit(1)
        if recall == 0:
            ratio=0
            hm=0
        else:
            if EntailClassifier.do_ratio:
                ratio = precision/recall
            else:
                ratio=precision-recall
                hm=2*precision*recall/(precision+recall)
        #print word1,word2,precision,recall,hm,ratio
        if ratio>threshold:
            predict =1
        else:
            predict =0
        return predict



    def svm_train(self,split,method):
        #generate positives and negatives for SVM
        X=[] #training points
        y=[] #classification of training points
        for [word1,word2,result] in self.pairmatrix[self.cv_idx!=split]:
            X.append(self.transform(word1,word2,method))
            y.append(result)

        #generate testpairs for SVM
        testpoints=[]
        actual=[]
        for [word1,word2,result] in self.pairmatrix[self.cv_idx==split]:
            testpoints.append(self.transform(word1,word2,method))
            actual.append(result)

        #build SVM
        print "Built training-test split "+str(split)+" ... building SVM"
        clf = svm.SVC()
        clf.fit(X,y)
        print "Built SVM"

        results=clf.predict(testpoints)
        #compare results against actual
        return f_analyse(actual,results)

    def transform(self,word1,word2,method):
        if method =="CR":
            return self.transform_CR(word1,word2)
        else:
            print "No such transformation method "+method

    def transform_CR(self,word1,word2):
        #return a vector of the recall precision comparison of two words for svm

        precision = float(self.entrydict[word1].precision(self.entrydict[word2]))
        recall=float(self.entrydict[word2].precision(self.entrydict[word1]))
        if recall==0:
            ratio=0
            hm=0
        else:
            ratio=precision/recall
            hm=2*precision*recall/(precision+recall)
        return [ratio,hm]

if __name__ == "__main__":
    parameters=conf.configure(sys.argv)
    starttime=time.time()
    print "Started at "+datetime.datetime.fromtimestamp(starttime).strftime('%Y-%m-%d %H:%M:%S')
    #print parameters

    myEntClassifier=EntailClassifier(parameters["pairfile"],parameters["freqfile"],parameters["use_cache"])

    #myEntClassifier.test1(0,"freq",[0])

    if "lin_freq" in parameters["methods"] or "lin" in parameters["methods"]:
        #need to load up thesaurus similarity file
        #probably want to cache the relevant similarity scores
        myEntClassifier.loadsims(parameters["simsfile"],parameters["use_cache"])
        #print myEntClassifier.entrydict["ambulance"].simdict
        #print myEntClassifier.entrydict["ambulance"].rankdict

    if "CR" in parameters["methods"] or "CR_thresh" in parameters["methods"] or "clarke" in parameters["methods"] or "clarke_thresh" in parameters["methods"] or "invCL" in parameters["methods"]:
        myEntClassifier.loadvectors(parameters["vectorfile"],parameters["use_cache"])


    for method in parameters["methods"]:
        myEntClassifier.traintest(method)


    endtime=time.time()
    print "Finished at "+datetime.datetime.fromtimestamp(endtime).strftime('%Y-%m-%d %H:%M:%S')
    elapsed = endtime-starttime
    hourselapsed=int(elapsed/3600)
    minselapsed=int((elapsed-hourselapsed*3600)/60)
    print "Time taken is "+str(hourselapsed)+" hours and "+str(minselapsed)+" minutes."
