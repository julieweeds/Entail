__author__ = 'Julie'
from mysvm import MySVM

import unittest

class SVMTest(unittest.TestCase):
    def test_reformat(self):

        negatives=[[3,1],[-4,-5],[10,-1]]
        positives=[[3,6],[-1,0],[2,3]]
        (X,y)=MySVM.reformat(positives,negatives)
        self.assertEqual(y,[1,1,1,0,0,0])
        self.assertEqual(X,[[3,6],[-1,0],[2,3],[3,1],[-4,-5],[10,-1]])

    def test_easy(self):
        #y>x
        negatives=[[3,1],[-4,-5],[10,-1]]
        positives=[[3,6],[-1,0],[2,3]]

        testpairs=[[3,1],[3,6],[3,2],[3,4],[3,2.2],[3,3.5]]
        testanswers=[0,1,0,1,0,1]
        results = MySVM.train_test(positives,negatives,testpairs)
        for i in range(len(testpairs)):
            print "Test "+str(i)+" : "+str(testanswers[i])+" : "+str(results[i])
            self.assertEqual(results[i],testanswers[i])

   # def test_level2(self):
   #     negatives = [-8,-5,-3,2,21]
   #     positives = [-1,6,12,15,20]
   #     result = MySVM.classify(positives,negatives)
   #     #threshold needs to be 2<=t<6
   #     self.assertLess(result,6)
   #     self.assertGreaterEqual(result,2)

