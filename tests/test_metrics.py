from condor_tensorflow.metrics import OrdinalMeanAbsoluteError
from condor_tensorflow.metrics import SparseOrdinalMeanAbsoluteError
from condor_tensorflow.metrics import OrdinalAccuracy
from condor_tensorflow.metrics import SparseOrdinalAccuracy
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

def test_OrdinalMeanAbsoluteError1():
    loss = OrdinalMeanAbsoluteError()
    val = loss(tf.constant([[1., 0.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalMeanAbsoluteError1():
    loss = SparseOrdinalMeanAbsoluteError()
    val = loss(tf.constant([1]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_OrdinalMeanAbsoluteError2():
    loss = OrdinalMeanAbsoluteError()
    val = loss(tf.constant([[0., 0.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalMeanAbsoluteError2():
    loss = SparseOrdinalMeanAbsoluteError()
    val = loss(tf.constant([0]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_OrdinalAccuracy():
    loss = OrdinalAccuracy()
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalAccuracy():
    loss = SparseOrdinalAccuracy()
    val = loss(tf.constant([2]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_OrdinalAccuracy1():
    loss = OrdinalAccuracy(tolerance=1)
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalAccuracy1():
    loss = SparseOrdinalAccuracy(tolerance=1)
    val = loss(tf.constant([2]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_OrdinalAccuracy2():
    loss = OrdinalAccuracy()
    val = loss(tf.constant([[1., 0.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalAccuracy2():
    loss = SparseOrdinalAccuracy()
    val = loss(tf.constant([1]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_OrdinalAccuracy12():
    loss = OrdinalAccuracy(tolerance=1)
    val = loss(tf.constant([[1., 0.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalAccuracy12():
    loss = SparseOrdinalAccuracy(tolerance=1)
    val = loss(tf.constant([1]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)
