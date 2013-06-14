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
