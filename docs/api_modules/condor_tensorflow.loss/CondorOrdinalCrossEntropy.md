## CondorOrdinalCrossEntropy

*CondorOrdinalCrossEntropy(num_classes, importance_weights=None, from_type='ordinal_logits', labels_encoded=True, name='ordinal_crossent', **kwargs)*

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

### Properties

