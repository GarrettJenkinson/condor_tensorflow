from condor_tensorflow.activations import ordinal_softmax
import pytest
import tensorflow as tf


def test_ordinal_softmax():
    x = tf.constant([[-1.,1],[-2,2]])
    res = ordinal_softmax(x)
    expect = tf.constant([[0.7310586 , 0.07232949, 0.19661194],
                          [0.8807971 , 0.01420934, 0.10499357]], dtype=tf.float32)
    tf.debugging.assert_near(res,expect,rtol=1e-5,atol=1e-5)
