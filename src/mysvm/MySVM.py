__author__ = 'Julie'


from sklearn import svm

def reformat(positives,negatives):
    X=[]
    y=[]
    for list in positives:
        X.append(list)
        y.append(1)
    for list in negatives:
        X.append(list)
        y.append(0)
    return (X,y)

def train_test(positives,negatives,testpairs):

    (X,y)=reformat(positives,negatives)
    results=[0]
    clf = svm.SVC()
    print clf.fit(X,y)

    results=clf.predict(testpairs)
    return results

