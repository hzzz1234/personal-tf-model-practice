from unittest import mock
import unittest

class Count():

    def add(self,a,b):
        return a+b

# test Count class
class TestCount(unittest.TestCase):

    def test_add(self):
        count = Count()
        count.add = mock.Mock(return_value=13,side_effect=count.add)
        result = count.add(8,4)
        self.assertEqual(result,13)


if __name__ == '__main__':
    unittest.main()