from sep import Separator

__author__ = 'juliewe'
import unittest

class SeparatorTest(unittest.TestCase):
    def test_separated_lists(self):
        negatives=[-4,2,3]
        positives=[5,10,12]

        result = Separator.separate(positives,negatives)
        self.assertEqual(result,4)

    def test_overlap(self):
        negatives = [-8,-5,-3,2,21]
        positives = [-1,6,12,15,20]
        result = Separator.separate(positives,negatives)
        #threshold needs to be 2<=t<6
        self.assertLess(result,6)
        self.assertGreaterEqual(result,2)

