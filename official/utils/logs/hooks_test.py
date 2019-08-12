from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf  # pylint: disable=g-bad-import-order
from official.utils.testing import mock_lib


class SquareTest(tf.test.TestCase):
    # 测试用例开始
    def setUp(self):
        """Mock out logging calls to verify if correct info is being monitored."""
        self._logger = mock_lib.MockBenchmarkLogger()

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.train.create_global_step()
            self.train_op = tf.assign_add(tf.train.get_global_step(), 1)
            self.global_step = tf.train.get_global_step()

    def test_examples_per_sec_every_1_steps(self):
        with self.graph.as_default():
            self._validate_log_every_n_steps(1, 0)
    def testSquare(self):
        with self.session():
            # 平方操作
            x = tf.square([2, 3])
            # 测试x的值是否等于[4,9]
            self.assertAllEqual(x.eval(), [4, 9])

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.test.main()