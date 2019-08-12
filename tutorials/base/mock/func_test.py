from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from unittest.mock import patch
from tutorials.base.mock import function



class MyTestCase(unittest.TestCase):

    @patch("function.multiply")
    def test_add_and_multiply2(self, mock_multiply):
        x = 3
        y = 5
        mock_multiply.return_value = 15
        addition, multiple = function.add_and_multiply(x, y)
        mock_multiply.assert_called_once_with(3, 5)

        self.assertEqual(8, addition)
        self.assertEqual(15, multiple)


if __name__ == "__main__":
    unittest.main()