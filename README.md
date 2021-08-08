<img src="docs/img/condor.png" width=300>

# Condor Ordinal regression in Tensorflow Keras

Tensorflow Keras implementation of Condor Ordinal Regression (aka ordinal classification) by Garrett Jenkinson (2021).

It is compatible with any state-of-the-art deep neural network architecture, 
requiring only modification of the output layer, the labels, the loss function.

We also have condor [implemented for pytorch](https://github.com/GarrettJenkinson/condor_pytorch).

This package includes:

  * Ordinal tensorflow loss function: `CondorOrdinalCrossEntropy`
  * Ordinal tensorflow error metric: `OrdinalMeanAbsoluteError`
  * Ordinal tensorflow error metric: `OrdinalEarthMoversDistance`
  * Ordinal tensorflow sparse loss function: `CondorSparseOrdinalCrossEntropy`
  * Ordinal tensorflow sparse error metric: `SparseOrdinalMeanAbsoluteError`
  * Ordinal tensorflow sparse error metric: `SparseOrdinalEarthMoversDistance`
  * Ordinal tensorflow activation function: `ordinal_softmax`
  * Ordinal sklearn label encoder: `CondorOrdinalEncoder`

## Installation

Install the stable version via pip:

```bash
pip install condor_tensorflow
```

Install the most recent code on GitHub via pip:

```bash
pip install git+https://github.com/GarrettJenkinson/condor_tensorflow/
```

## Dependencies

This package relies on Python 3.6+, Tensorflow 2.2+, sklearn, and numpy.

## Example

This is a quick example to show a basic model implementation. With actual 
data one would also want to specify the input shape.

```python
import condor_tensorflow as condor
NUM_CLASSES = 5
# Ordinal 'labels' variable has 5 labels, 0 through 4.
enc_labs = condor.CondorOrdinalEncoder(nclasses=NUM_CLASSES).fit_transform(labels)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, activation = "relu"))
model.add(tf.keras.layers.Dense(NUM_CLASSES-1)) # Note the "-1"
model.compile(loss = condor.CondorOrdinalCrossEntropy(num_classes = NUM_CLASSES),
              metrics = [condor.MeanAbsoluteErrorLabels()])
model.fit(x = X, y = enc_labs)
```

[See this colab notebook](https://colab.research.google.com/drive/) for extended examples of ordinal regression with Amazon reviews (universal sentence encoder).

Please post any issues to the [issue queue](https://github.com/GarrettJenkinson/condor_tensorflow/issues). 

**Acknowledgments**: Many thanks to [the coral ordinal tensorflow authors](https://github.com/ck37/coral-ordinal) whose repo was a roadmap for this codebase.

Key pending items:

  * Function docstrings
  * Docs
  * Tests

## References

TBD.
