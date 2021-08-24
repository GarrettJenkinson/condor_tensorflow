from condor_tensorflow.labelencoder import CondorOrdinalEncoder
import pytest
import tensorflow as tf
import numpy as np
import pandas as pd


def test_labelencoder_basic():
    NUM_CLASSES = 5
    labels = np.arange(NUM_CLASSES)
    enc_labs = CondorOrdinalEncoder(nclasses=NUM_CLASSES).fit_transform(labels)
    expected = np.array([[0., 0., 0., 0.],
                         [1., 0., 0., 0.],
                         [1., 1., 0., 0.],
                         [1., 1., 1., 0.],
                         [1., 1., 1., 1.]])
    np.testing.assert_allclose(enc_labs, expected)


def test_labelencoder_advanced1():
    labels = np.array(['a', 'b', 'c', 'd', 'e'])
    enc_labs = CondorOrdinalEncoder().fit_transform(labels)
    expected = np.array([[0., 0., 0., 0.],
                         [1., 0., 0., 0.],
                         [1., 1., 0., 0.],
                         [1., 1., 1., 0.],
                         [1., 1., 1., 1.]])
    np.testing.assert_allclose(enc_labs, expected)


def test_labelencoder_advanced2():
    labels = ['a', 'b', 'c', 'd', 'e']
    enc_labs = CondorOrdinalEncoder().fit_transform(labels)
    expected = np.array([[0., 0., 0., 0.],
                         [1., 0., 0., 0.],
                         [1., 1., 0., 0.],
                         [1., 1., 1., 0.],
                         [1., 1., 1., 1.]])
    np.testing.assert_allclose(enc_labs, expected)


def test_labelencoder_advanced3():
    labels = ['low', 'med', 'high']
    enc = CondorOrdinalEncoder(categories=[['low', 'med', 'high']])
    enc_labs = enc.fit_transform(labels)
    expected = np.array([[0., 0.],
                         [1., 0.],
                         [1., 1.]])
    np.testing.assert_allclose(enc_labs, expected)
