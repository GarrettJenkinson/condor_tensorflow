from condor_tensorflow.metrics import OrdinalMeanAbsoluteError
from condor_tensorflow.metrics import SparseOrdinalMeanAbsoluteError
from condor_tensorflow.metrics import OrdinalEarthMoversDistance
from condor_tensorflow.metrics import SparseOrdinalEarthMoversDistance
import pytest
import tensorflow as tf


def test_OrdinalMeanAbsoluteError():
    loss = OrdinalMeanAbsoluteError()
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(2.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseOrdinalMeanAbsoluteError():
    loss = SparseOrdinalMeanAbsoluteError()
    val = loss(tf.constant([2]), tf.constant([[-1., 1.]]))
    expect = tf.constant(2.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_OrdinalEarthMoversDistance():
    loss = OrdinalEarthMoversDistance()
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.5344467)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseOrdinalEarthMoversDistance():
    loss = SparseOrdinalEarthMoversDistance()
    val = loss(tf.constant([2]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.5344467)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)
