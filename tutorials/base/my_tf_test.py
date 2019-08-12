from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf  # pylint: disable=g-bad-import-order

class SquareTest(tf.test.TestCase):
    # 测试用例开始
    def setUp(self):
        print("start")


    def testSquare(self):
        with self.session():
            # 平方操作
            x = tf.square([2, 3])
            # 测试x的值是否等于[4,9]
            self.assertAllEqual(x.eval(), [4, 9])

class TestHook(tf.estimator.LoggingTensorHook):
    def begin(self):
        print("begin")

    def after_run(self, unused_run_context, run_values):
        print("after_run")

    def end(self, session):
        print("end")



if __name__ == "__main__":
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    tf.test.main()