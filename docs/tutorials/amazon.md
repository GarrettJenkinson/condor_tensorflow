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
print("CONDOR Ordinal version:", condor.__version__)
```

## Amazon reviews and 5-star ratings

Amazon review data via https://nijianmo.github.io/amazon/index.html#subsets


```
!curl -o Prime_Pantry_5.json.gz http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Prime_Pantry_5.json.gz 
```


```python
data = []
with gzip.open('Prime_Pantry_5.json.gz') as f:
    for l in f:
        data.append(json.loads(l.strip()))

df = pd.DataFrame.from_dict(data)
df = df[['overall', 'reviewText']]

# There is a large amount of duplicate text in here, possibly due to paid/fraudulent reviews.
df.drop_duplicates("reviewText", inplace = True)

# Some of the text is blank, which causes an obscure error about floating point conversion.
df.dropna(inplace = True)

print(len(df))
print(df.head())

outcome_col = "overall"
text_col = "reviewText"

# We subtract the minimum value from the outcomes so that they start at 0.
df[outcome_col] = df[outcome_col].values - df[outcome_col].min()

print("\n", df.overall.value_counts())

# TODO: define automatically based on the number of unique values in the outcome variable.
num_classes = 5
```


```python
# Train/Test split
text_train, text_test, labels_train, labels_test = \
  train_test_split(df[text_col].values, df[outcome_col].values, test_size = 10000, random_state = 1)

print("Training text shape:", text_train.shape)
print("Training labels shape:", labels_train.shape)
print("Testing text shape:", text_test.shape)
print("Testing labels shape:", labels_test.shape)
```

### Universal Sentence Encoder model (minimal code changes)


```python
# This takes 20 - 30 seconds.

# Clear our GPU memory to stay efficient.
tf.keras.backend.clear_session()

input_text = tf.keras.layers.Input(shape = [], dtype = tf.string, name = 'input_text')

model_url = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

base_model = hub.KerasLayer(model_url, input_shape = [],
                            dtype = tf.string,
                            trainable = False)
                            
embedded = base_model(input_text)

x = tf.keras.layers.Dense(64, activation = 'relu')(embedded)
x = tf.keras.layers.Dropout(0.1)(x)
output =tf.keras.layers.Dense(num_classes-1)(x) 

model = tf.keras.Model(inputs = input_text, outputs = output)

model.summary()
```


```python
model.compile(loss = condor.SparseCondorOrdinalCrossEntropy(),
              metrics = [condor.SparseOrdinalEarthMoversDistance(),
                         condor.SparseOrdinalMeanAbsoluteError()],
              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))
```


```python
# Encode a test string and take a look at the first ten dimensions.
base_model(np.array(["test_string"])).numpy()[0, :10]
```


```python
history = model.fit(x = text_train,
                    y = labels_train,
                    epochs = 5,
                    batch_size = 32, 
                    validation_split = 0.2,
                    callbacks = [tf.keras.callbacks.EarlyStopping(patience = 2,
                                                                  min_delta = 0.001,
                                                                  restore_best_weights = True)])
```

#### Evaluate


```python
# For comparison, CORAL achieves loss 0.7962, MAE 0.3195
model.evaluate(text_test, labels_test) 
```


```python
# Generate predictions - initially these are cumulative logits.
preds = model.predict(text_test)
print(preds)
# Convert cumulative logits to probabilities for each class aka rank or label.
probs = pd.DataFrame(condor.ordinal_softmax(preds).numpy())
```


```python
print(probs.head(10))
print(labels_test[:10])
```

#### Evaluate accuracy


```python
# Evaluate accuracy and mean absolute error
labels_v1 = probs.idxmax(axis = 1)
print("Accuracy of label version 1:", np.mean(labels_v1 == labels_test))

# Compare to logit-based cumulative probs
cum_probs = pd.DataFrame(preds).apply(special.expit).cumprod(axis=1)
# Calculate the labels using the style of Cao et al.
labels_v2 = cum_probs.apply(lambda x: x > 0.5).sum(axis = 1)
print("Accuracy of label version 2:", np.mean(labels_v2 == labels_test))
```

#### Evaluate mean absolute label error

This is effectively an ordinal version of 1 - accuracy.


```python
# These do not correspond with what we get from the model evaluation. Something must be off in one of these.
print("Mean absolute label error version 1:", np.mean(np.abs(labels_v1 - labels_test)))
print("Mean absolute label error version 2:", np.mean(np.abs(labels_v2 - labels_test)))

print("Root mean squared label error version 1:", np.sqrt(np.mean(np.square(labels_v1 - labels_test))))
print("Root mean squared label error version 2:", np.sqrt(np.mean(np.square(labels_v2 - labels_test))))
```


```python
# Review how absolute error is calculated for ordinal labels:
pd.DataFrame({"true": labels_test, "pred_v2": labels_v1, "abs": labels_v2 - labels_test}).head()
```

### Universal Sentence Encoder model (speed up using encodings)

The "Sparse" versions of the CONDOR API are convenient and require minimal code changes. However there is a performance overhead compared to if we pre-encode the labels using CONDORs ordinal encoder. The sparse API is basically encoding on the fly inside the training loop. 

Also as we will see later, the labels do not always come encoded as 0,1,...,K-1. In this case, using the CondorOrdinalEncoder will help transform labels into ordinal-ready values.


```python
# pre-encoding runs very fast so the savings later are worth it
enc = condor.CondorOrdinalEncoder(nclasses=num_classes)
enc_labs_train = enc.fit_transform(labels_train)
enc_labs_test = enc.transform(labels_test)
```


```python
# Note the lack of "Sparse" in the condor functions here
model.compile(loss = condor.CondorOrdinalCrossEntropy(),
              metrics = [condor.OrdinalEarthMoversDistance(),
                         condor.OrdinalMeanAbsoluteError()],
              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))
```


```python
# note the encoded labels are passed to the fit now
history = model.fit(x = text_train,
                    y = enc_labs_train,
                    epochs = 5,
                    batch_size = 32, 
                    validation_split = 0.2,
                    callbacks = [tf.keras.callbacks.EarlyStopping(patience = 2,
                                                                  min_delta = 0.001,
                                                                  restore_best_weights = True)])
```


```python
model.evaluate(text_test, enc_labs_test) 
```

#### More examples of label encoding capabilities
Here we demo the features of the ordinal encoder.



```python
# Here the ordinal encoder figures out how many classes there are automatically
# and orders them in the default sklearn OrdinalEncoder fashion 
# (i.e., alphabetically here)
labels = np.array(['a','b','c','d','e'])
enc_labs = condor.CondorOrdinalEncoder().fit_transform(labels)
print(enc_labs)
```


```python
# Here the ordinal encoder figures out how many classes there are automatically
# and orders them in the default sklearn OrdinalEncoder fashion 
# (i.e., alphabetically here). This time it is dealing with a basic list.
labels = ['a','b','c','d','e']
enc_labs = condor.CondorOrdinalEncoder().fit_transform(labels)

print(enc_labs)
```


```python
# Here we wish to specify the order to be different from alphabetical. Note
# this would also allow "missing" categories to be included in proper order.
labels = ['low','med','high']
enc = condor.CondorOrdinalEncoder(categories=[['low', 'med', 'high']])
enc_labs = enc.fit_transform(labels)

print(enc_labs)
```
