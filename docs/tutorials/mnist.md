# CONDOR Ordinal classification/regression in Tensorflow Keras 


## Import statements


```python
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy import special
import tensorflow_hub as hub
import os
import json
import gzip
from urllib.request import urlopen

import tensorflow as tf
print("Tensorflow version", tf.__version__)

import condor_tensorflow as condor
print("CORAL Ordinal version:", condor.__version__)
```

## MNIST toy example

This outcome is not actually ordinal, it's categorical. We're just using it as a toy example to show how the different components are used.


```python
# Hyperparameters
random_seed = 1 # Not yet used
learning_rate = 0.05
batch_size = 128
num_epochs = 2

# Architecture
NUM_CLASSES = 10
```


```python
# Fetch and format the mnist data
(mnist_images, mnist_labels), (mnist_images_test, mnist_labels_test) = tf.keras.datasets.mnist.load_data()

# Split off a validation dataset for early stopping
mnist_images, mnist_images_val, mnist_labels, mnist_labels_val = \
  model_selection.train_test_split(mnist_images, mnist_labels, test_size = 5000, random_state = 1)

print("Shape of training images:", mnist_images.shape)
print("Shape of training labels:", mnist_labels.shape)

print("Shape of test images:", mnist_images_test.shape)
print("Shape of test labels:", mnist_labels_test.shape)

print("Shape of validation images:", mnist_images_val.shape)
print("Shape of validation labels:", mnist_labels_val.shape)

# Also rescales to 0-1 range.
dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images[..., tf.newaxis] / 255, tf.float32),
   tf.cast(mnist_labels, tf.int64)))
dataset = dataset.shuffle(1000).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images_test[..., tf.newaxis] / 255, tf.float32),
   tf.cast(mnist_labels_test, tf.int64)))
#test_dataset = test_dataset.shuffle(1000).batch(batch_size)
# Here we do not shuffle the test dataset.
test_dataset = test_dataset.batch(batch_size)


val_dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images_val[..., tf.newaxis] / 255, tf.float32),
   tf.cast(mnist_labels_val, tf.int64)))
val_dataset = val_dataset.shuffle(1000).batch(batch_size)
```

### Simple MLP model



Now we create a simple multi-layer perceptron model so that we can apply the ordinal output layer.


```python
def create_model(num_classes):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Flatten(input_shape = (28, 28, )))
  model.add(tf.keras.layers.Dense(128, activation = "relu"))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.Dense(32, activation = "relu"))
  model.add(tf.keras.layers.Dropout(0.1))
  # No activation function specified so this will output cumulative logits.
  model.add(tf.keras.layers.Dense(num_classes-1))
  return model

model = create_model(NUM_CLASSES)

# Note that the model generates 1 fewer outputs than the number of classes. 
model.summary()
```


```python
# Or a functional API version
def create_model2(num_classes):
  inputs = tf.keras.Input(shape = (28, 28, ))

  x = tf.keras.layers.Flatten()(inputs)
  x = tf.keras.layers.Dense(128, activation = "relu")(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(32, activation = "relu")(x)
  x = tf.keras.layers.Dropout(0.1)(x)
  # No activation function specified so this will output cumulative logits.
  outputs = tf.keras.layers.Dense(num_classes-1)(x)

  model = tf.keras.Model(inputs = inputs, outputs = outputs)

  return model

model = create_model2(NUM_CLASSES)

# Note that the model generates 1 fewer outputs than the number of classes. 
model.summary()
```


```python
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
              loss = condor.SparseCondorOrdinalCrossEntropy(),
              metrics = [condor.SparseOrdinalEarthMoversDistance(),
                         condor.SparseOrdinalMeanAbsoluteError()])
```


```python
history = model.fit(dataset, epochs = 5, validation_data = val_dataset,
                    callbacks = [tf.keras.callbacks.EarlyStopping(patience = 3, restore_best_weights = True)])
```

### Test set evaluation


```python
# Evaluate on test dataset.
model.evaluate(test_dataset)
```

### Cumulative logits to probabilities

We can convert the cumulative logit output of the layer into the probability estimate for each ordinal label. This can then be used to calculate other metrics like accuracy or mean absolute error.

Notice that the probability distribution for each observation is unimodal, which is what we want for an ordinal outcome variable.


```python
print("Predict on test dataset")

# Note that these are ordinal (cumulative) logits, not probabilities or regular logits.
ordinal_logits = model.predict(test_dataset)

# Convert from logits to label probabilities. This is initially a tensorflow tensor.
tensor_probs = condor.ordinal_softmax(ordinal_logits)

# Convert the tensor into a pandas dataframe.
probs_df = pd.DataFrame(tensor_probs.numpy())

probs_df.head()
```


```python
# Check that probabilities all sum to 1 - looks good!
probs_df.sum(axis = 1)
```

### Label prediction

This notebook shows two ways of calculating predicted labels. We can take the highest probability label (first method) or we can choose the highest label with Pr(Y > label) > 50%.


```python
# Probs to labels
labels = probs_df.idxmax(axis = 1)
labels.values
```


```python
# What is our accuracy? Around 69%.
np.mean(labels == mnist_labels_test)
```


```python
# Compare to logit-based cumulative probs
cum_probs = pd.DataFrame(ordinal_logits).apply(special.expit).cumprod(axis=1)
cum_probs.head()
```

Now we should try another option, which is used in the Cao et al. paper.


```python
# Calculate the labels using the style of Cao et al.
labels2 = cum_probs.apply(lambda x: x > 0.5).sum(axis = 1)
labels2.head()
```


```python
# What is the accuracy of these labels? 
np.mean(labels2 == mnist_labels_test)
```


```python
# More often than not these are the same, but still a lot of discrepancy.
np.mean(labels == labels2)
```


```python
print("Mean absolute label error version 1:", np.mean(np.abs(labels - mnist_labels_test)))
print("Mean absolute label error version 2:", np.mean(np.abs(labels2 - mnist_labels_test)))
```


```python
mnist_labels_test[:5]
```

### Importance weights customization

A quick example to show how the importance weights can be customized. 


```python
model = create_model(num_classes = NUM_CLASSES)
model.summary()

# We have num_classes - 1 outputs (cumulative logits), so there are 9 elements
# in the importance vector to customize.
importance_weights = [1., 1., 0.5, 0.5, 0.5, 1., 1., 0.1, 0.1]
loss_fn = condor.SparseCondorOrdinalCrossEntropy(importance_weights = importance_weights)

model.compile(tf.keras.optimizers.Adam(lr = learning_rate), loss = loss_fn)
```


```python
history = model.fit(dataset, epochs = num_epochs)
```

