__author__ = 'Julie'

import re,math

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

def f_analyse(actual,results):
    #return acc,p,r,f for 2 lists

    TP=0
    TN=0
    FP=0
    FN=0

    #check lengths the same
    if len(actual)!=len(results):
        print "Error: two lists not of same length: "+str(len(actual))+", "+str(len(results))

    test=0
    while len(actual)>0:
        test+=1
        thisactual=actual.pop()
        thisresult=results.pop()

        if thisactual==1:
            if thisresult==1:
                TP+=1
            else:
                FN+=1
        else:
            if thisresult==1:
                FP+=1
            else:
                TN+=1

    total=TP+FP+TN+FN
    if total!=test:
        print "Error: number of tests was "+str(test)+" but total is "+str(total)
    acc=float(TP+TN)/float(total)
    pre=float(TP)/float(TP+FP)
    rec=float(TP)/float(TP+FN)
    f=2*pre*rec/(pre+rec)
    return (acc,pre,rec,f)


if __name__ =="__main__":

    a1=[1,1,1,1,0,0,0,0]
    r1=[1,1,0,0,1,1,0,0]

    print f_analyse(a1,r1)