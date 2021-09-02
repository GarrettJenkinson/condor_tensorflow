condor_tensorflow version: 0.1.0-dev
## SparseCondorOrdinalCrossEntropy

*SparseCondorOrdinalCrossEntropy(importance_weights=None, from_type='ordinal_logits', name='ordinal_crossent', **kwargs)*

Loss base class.

    To be implemented by subclasses:
    * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.

    Example subclass implementation:

    ```python
    class MeanSquaredError(Loss):

    def call(self, y_true, y_pred):
    y_pred = tf.convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(math_ops.square(y_pred - y_true), axis=-1)
    ```

    When used with `tf.distribute.Strategy`, outside of built-in training loops
    such as `tf.keras` `compile` and `fit`, please use 'SUM' or 'NONE' reduction
    types, and reduce losses explicitly in your training loop. Using 'AUTO' or
    'SUM_OVER_BATCH_SIZE' will raise an error.

    Please see this custom training [tutorial](
    https://www.tensorflow.org/tutorials/distribute/custom_training) for more
    details on this.

    You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:
    ```python
    with strategy.scope():
    loss_obj = tf.keras.losses.CategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE)
    ....
    loss = (tf.reduce_sum(loss_obj(labels, predictions)) *
    (1. / global_batch_size))
    ```

### Methods

<hr>

*call(y_true, y_pred)*

Invokes the `Loss` instance.

    Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
    sparse loss functions such as sparse categorical crossentropy where
    shape = `[batch_size, d0, .. dN-1]`
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`

    Returns:
    Loss values with the shape `[batch_size, d0, .. dN-1]`.

<hr>

*from_config(config)*

Instantiates a `Loss` from its config (output of `get_config()`).

    Args:
    config: Output of `get_config()`.

    Returns:
    A `Loss` instance.

<hr>

*get_config()*

Returns the config dictionary for a `Loss` instance.

<hr>

*label_to_levels(label)*

None

<hr>

*ordinal_loss(logits, levels, importance)*

Cross-entropy loss function designed for ordinal outcomes.

**Parameters**

logits: tf.Tensor, shape=(num_samples,num_classes-1)
    Logit output of the final Dense(num_classes-1) layer.

    levels: tf.Tensor, shape=(num_samples, num_classes-1)
    Encoded lables provided by CondorOrdinalEncoder.

    importance_weights: tf or np array of floats, shape(numclasses-1,)
    Importance weights for each binary classification task.

**Returns**

loss: tf.Tensor, shape=(num_samples,)
    Loss vector, note that tensorflow will reduce it to a single number
    automatically.

### Properties

## SparseOrdinalEarthMoversDistance

*SparseOrdinalEarthMoversDistance(**kwargs)*

Computes earth movers distance for ordinal labels.

### Methods

<hr>

*call(y_true, y_pred)*

Computes mean absolute error for ordinal labels.

    Args:
    y_true: Cumulatiuve logits from CondorOrdinal layer.
    y_pred: Sparse Labels with values in {0,1,...,num_classes-1}
    sample_weight (optional): Not implemented.

<hr>

*from_config(config)*

Instantiates a `Loss` from its config (output of `get_config()`).

    Args:
    config: Output of `get_config()`.

    Returns:
    A `Loss` instance.

<hr>

*get_config()*

Returns the serializable config of the metric.

### Properties

## CondorOrdinalCrossEntropy

*CondorOrdinalCrossEntropy(importance_weights=None, from_type='ordinal_logits', name='ordinal_crossent', **kwargs)*

Loss base class.

    To be implemented by subclasses:
    * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.

    Example subclass implementation:

    ```python
    class MeanSquaredError(Loss):

    def call(self, y_true, y_pred):
    y_pred = tf.convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(math_ops.square(y_pred - y_true), axis=-1)
    ```

    When used with `tf.distribute.Strategy`, outside of built-in training loops
    such as `tf.keras` `compile` and `fit`, please use 'SUM' or 'NONE' reduction
    types, and reduce losses explicitly in your training loop. Using 'AUTO' or
    'SUM_OVER_BATCH_SIZE' will raise an error.

    Please see this custom training [tutorial](
    https://www.tensorflow.org/tutorials/distribute/custom_training) for more
    details on this.

    You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:
    ```python
    with strategy.scope():
    loss_obj = tf.keras.losses.CategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE)
    ....
    loss = (tf.reduce_sum(loss_obj(labels, predictions)) *
    (1. / global_batch_size))
    ```

### Methods

<hr>

*call(y_true, y_pred)*

Invokes the `Loss` instance.

    Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
    sparse loss functions such as sparse categorical crossentropy where
    shape = `[batch_size, d0, .. dN-1]`
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`

    Returns:
    Loss values with the shape `[batch_size, d0, .. dN-1]`.

<hr>

*from_config(config)*

Instantiates a `Loss` from its config (output of `get_config()`).

    Args:
    config: Output of `get_config()`.

    Returns:
    A `Loss` instance.

<hr>

*get_config()*

Returns the config dictionary for a `Loss` instance.

<hr>

*ordinal_loss(logits, levels, importance)*

Cross-entropy loss function designed for ordinal outcomes.

**Parameters**

logits: tf.Tensor, shape=(num_samples,num_classes-1)
    Logit output of the final Dense(num_classes-1) layer.

    levels: tf.Tensor, shape=(num_samples, num_classes-1)
    Encoded lables provided by CondorOrdinalEncoder.

    importance_weights: tf or np array of floats, shape(numclasses-1,)
    Importance weights for each binary classification task.

**Returns**

loss: tf.Tensor, shape=(num_samples,)
    Loss vector, note that tensorflow will reduce it to a single number
    automatically.

### Properties

## OrdinalEarthMoversDistance

*OrdinalEarthMoversDistance(name='earth_movers_distance', **kwargs)*

Computes earth movers distance for ordinal labels.

### Methods

<hr>

*call(y_true, y_pred)*

Computes mean absolute error for ordinal labels.

    Args:
    y_true: Cumulatiuve logits from CondorOrdinal layer.
    y_pred: CondorOrdinal Encoded Labels.
    sample_weight (optional): Not implemented.

<hr>

*from_config(config)*

Instantiates a `Loss` from its config (output of `get_config()`).

    Args:
    config: Output of `get_config()`.

    Returns:
    A `Loss` instance.

<hr>

*get_config()*

Returns the serializable config of the metric.

### Properties

