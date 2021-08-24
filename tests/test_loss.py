from condor_tensorflow.loss import CondorOrdinalCrossEntropy
from condor_tensorflow.loss import SparseCondorOrdinalCrossEntropy
import pytest
import tensorflow as tf


def test_CondorOrdinalCrossEntropy():
    loss = CondorOrdinalCrossEntropy()
    val = loss(tf.constant([[-1., 1.]]), tf.constant([[1., 1.]]))
    expect = tf.constant(2.9397845)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseCondorOrdinalCrossEntropy():
    loss = SparseCondorOrdinalCrossEntropy()
    val = loss(tf.constant([2]), tf.constant([[-1., 1.]]))
    expect = tf.constant(2.9397845)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)
