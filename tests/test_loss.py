from condor_tensorflow.loss import CondorNegLogLikelihood
from condor_tensorflow.loss import SparseCondorNegLogLikelihood
from condor_tensorflow.loss import CondorOrdinalCrossEntropy
from condor_tensorflow.loss import SparseCondorOrdinalCrossEntropy
from condor_tensorflow.loss import OrdinalEarthMoversDistance
from condor_tensorflow.loss import SparseOrdinalEarthMoversDistance
import pytest
import tensorflow as tf


def test_CondorNegLogLikelihood():
    loss = CondorNegLogLikelihood()
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.6265235)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseCondorNegLogLikelihood():
    loss = SparseCondorNegLogLikelihood()
    val = loss(tf.constant([2]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.6265235)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_CondorOrdinalCrossEntropy():
    loss = CondorOrdinalCrossEntropy()
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(2.9397845)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseCondorOrdinalCrossEntropy():
    loss = SparseCondorOrdinalCrossEntropy()
    val = loss(tf.constant([2]), tf.constant([[-1., 1.]]))
    expect = tf.constant(2.9397845)
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
