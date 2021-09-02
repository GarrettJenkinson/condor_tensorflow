condor_tensorflow version: 0.1.0-dev
## OrdinalMeanAbsoluteError

*OrdinalMeanAbsoluteError(*args, **kwargs)*

Computes mean absolute error for ordinal labels.

### Methods

<hr>

*add_loss(losses, **kwargs)*

Add loss tensor(s), potentially dependent on layer inputs.

    Some losses (for instance, activity regularization losses) may be dependent
    on the inputs passed when calling a layer. Hence, when reusing the same
    layer on different inputs `a` and `b`, some entries in `layer.losses` may
    be dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    This method can be used inside a subclassed layer or model's `call`
    function, in which case `losses` should be a Tensor or list of Tensors.

    Example:

    ```python
    class MyLayer(tf.keras.layers.Layer):
    def call(self, inputs):
    self.add_loss(tf.abs(tf.reduce_mean(inputs)))
    return inputs
    ```

    This method can also be called directly on a Functional Model during
    construction. In this case, any loss Tensors passed to this Model must
    be symbolic and be able to be traced back to the model's `Input`s. These
    losses become part of the model's topology and are tracked in `get_config`.

    Example:

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    # Activity regularization.
    model.add_loss(tf.abs(tf.reduce_mean(x)))
    ```

    If this is not the case for your loss (if, for example, your loss references
    a `Variable` of one of the model's layers), you can wrap your loss in a
    zero-argument lambda. These losses are not tracked as part of the model's
    topology since they can't be serialized.

    Example:

    ```python
    inputs = tf.keras.Input(shape=(10,))
    d = tf.keras.layers.Dense(10)
    x = d(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    # Weight regularization.
    model.add_loss(lambda: tf.reduce_mean(d.kernel))
    ```

    Arguments:
    losses: Loss tensor, or list/tuple of tensors. Rather than tensors, losses
    may also be zero-argument callables which create a loss tensor.
**kwargs: Additional keyword arguments for backward compatibility.
    Accepted values:

inputs - Deprecated, will be automatically inferred.

<hr>

*add_metric(value, name=None, **kwargs)*

Adds metric tensor to the layer.

    This method can be used inside the `call()` method of a subclassed layer
    or model.

    ```python
    class MyMetricLayer(tf.keras.layers.Layer):
    def __init__(self):
    super(MyMetricLayer, self).__init__(name='my_metric_layer')
    self.mean = tf.keras.metrics.Mean(name='metric_1')

    def call(self, inputs):
    self.add_metric(self.mean(x))
    self.add_metric(tf.reduce_sum(x), name='metric_2')
    return inputs
    ```

    This method can also be called directly on a Functional Model during
    construction. In this case, any tensor passed to this Model must
    be symbolic and be able to be traced back to the model's `Input`s. These
    metrics become part of the model's topology and are tracked when you
    save the model via `save()`.

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.add_metric(math_ops.reduce_sum(x), name='metric_1')
    ```

    Note: Calling `add_metric()` with the result of a metric object on a
    Functional Model, as shown in the example below, is not supported. This is
    because we cannot trace the metric result tensor back to the model's inputs.

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.add_metric(tf.keras.metrics.Mean()(x), name='metric_1')
    ```

    Args:
    value: Metric tensor.
    name: String metric name.
**kwargs: Additional keyword arguments for backward compatibility.
    Accepted values:

`aggregation` - When the `value` tensor provided is not the result of
    calling a `keras.Metric` instance, it will be aggregated by default
    using a `keras.Metric.Mean`.

<hr>

*add_update(updates, inputs=None)*

Add update op(s), potentially dependent on layer inputs.

    Weight updates (for instance, the updates of the moving mean and variance
    in a BatchNormalization layer) may be dependent on the inputs passed
    when calling a layer. Hence, when reusing the same layer on
    different inputs `a` and `b`, some entries in `layer.updates` may be
    dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    This call is ignored when eager execution is enabled (in that case, variable
    updates are run on the fly and thus do not need to be tracked for later
    execution).

    Arguments:
    updates: Update op, or list/tuple of update ops, or zero-arg callable
    that returns an update op. A zero-arg callable should be passed in
    order to disable running the updates by setting `trainable=False`
    on this Layer, when executing in Eager mode.
    inputs: Deprecated, will be automatically inferred.

<hr>

*add_variable(*args, **kwargs)*

Deprecated, do NOT use! Alias for `add_weight`.

<hr>

*add_weight(name, shape=(), aggregation=<VariableAggregation.SUM: 1>, synchronization=<VariableSynchronization.ON_READ: 3>, initializer=None, dtype=None)*

Adds state variable. Only for use by subclasses.

<hr>

*apply(inputs, *args, **kwargs)*

Deprecated, do NOT use!

    This is an alias of `self.__call__`.

    Arguments:
    inputs: Input tensor(s).
    *args: additional positional arguments to be passed to `self.call`.
**kwargs: additional keyword arguments to be passed to `self.call`.


Returns:
    Output tensor(s).

<hr>

*build(input_shape)*

Creates the variables of the layer (optional, for subclass implementers).

    This is a method that implementers of subclasses of `Layer` or `Model`
    can override if they need a state-creation step in-between
    layer instantiation and layer call.

    This is typically used to create the weights of `Layer` subclasses.

    Arguments:
    input_shape: Instance of `TensorShape`, or list of instances of
    `TensorShape` if the layer expects a list of inputs
    (one instance per input).

<hr>

*call(inputs, **kwargs)*

This is where the layer's logic lives.

    Note here that `call()` method in `tf.keras` is little bit different
    from `keras` API. In `keras` API, you can pass support masking for
    layers as additional arguments. Whereas `tf.keras` has `compute_mask()`
    method to support masking.

    Arguments:
    inputs: Input tensor, or list/tuple of input tensors.
**kwargs: Additional keyword arguments. Currently unused.


Returns:
    A tensor or list/tuple of tensors.

<hr>

*compute_mask(inputs, mask=None)*

Computes an output mask tensor.

    Arguments:
    inputs: Tensor or list of tensors.
    mask: Tensor or list of tensors.

    Returns:
    None or a tensor (or list of tensors,
    one per output tensor of the layer).

<hr>

*compute_output_shape(input_shape)*

Computes the output shape of the layer.

    If the layer has not been built, this method will call `build` on the
    layer. This assumes that the layer will later be used with inputs that
    match the input shape provided here.

    Arguments:
    input_shape: Shape tuple (tuple of integers)
    or list of shape tuples (one per output tensor of the layer).
    Shape tuples can include None for free dimensions,
    instead of an integer.

    Returns:
    An input shape tuple.

<hr>

*compute_output_signature(input_signature)*

Compute the output tensor signature of the layer based on the inputs.

    Unlike a TensorShape object, a TensorSpec object contains both shape
    and dtype information for a tensor. This method allows layers to provide
    output dtype information if it is different from the input dtype.
    For any layer that doesn't implement this function,
    the framework will fall back to use `compute_output_shape`, and will
    assume that the output dtype matches the input dtype.

    Args:
    input_signature: Single TensorSpec or nested structure of TensorSpec
    objects, describing a candidate input for the layer.

    Returns:
    Single TensorSpec or nested structure of TensorSpec objects, describing
    how the layer would transform the provided input.

    Raises:
    TypeError: If input_signature contains a non-TensorSpec object.

<hr>

*count_params()*

Count the total number of scalars composing the weights.

    Returns:
    An integer count.

    Raises:
    ValueError: if the layer isn't yet built
    (in which case its weights aren't yet defined).

<hr>

*from_config(config)*

Creates a layer from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same layer from the config
    dictionary. It does not handle layer connectivity
    (handled by Network), nor weights (handled by `set_weights`).

    Arguments:
    config: A Python dictionary, typically the
    output of get_config.

    Returns:
    A layer instance.

<hr>

*get_config()*

Returns the serializable config of the metric.

<hr>

*get_input_at(node_index)*

Retrieves the input tensor(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A tensor (or list of tensors if the layer has multiple inputs).

    Raises:
    RuntimeError: If called in Eager mode.

<hr>

*get_input_mask_at(node_index)*

Retrieves the input mask tensor(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A mask tensor
    (or list of tensors if the layer has multiple inputs).

<hr>

*get_input_shape_at(node_index)*

Retrieves the input shape(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A shape tuple
    (or list of shape tuples if the layer has multiple inputs).

    Raises:
    RuntimeError: If called in Eager mode.

<hr>

*get_losses_for(inputs)*

Deprecated, do NOT use!

    Retrieves losses relevant to a specific set of inputs.

    Arguments:
    inputs: Input tensor or list/tuple of input tensors.

    Returns:
    List of loss tensors of the layer that depend on `inputs`.

<hr>

*get_output_at(node_index)*

Retrieves the output tensor(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A tensor (or list of tensors if the layer has multiple outputs).

    Raises:
    RuntimeError: If called in Eager mode.

<hr>

*get_output_mask_at(node_index)*

Retrieves the output mask tensor(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A mask tensor
    (or list of tensors if the layer has multiple outputs).

<hr>

*get_output_shape_at(node_index)*

Retrieves the output shape(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A shape tuple
    (or list of shape tuples if the layer has multiple outputs).

    Raises:
    RuntimeError: If called in Eager mode.

<hr>

*get_updates_for(inputs)*

Deprecated, do NOT use!

    Retrieves updates relevant to a specific set of inputs.

    Arguments:
    inputs: Input tensor or list/tuple of input tensors.

    Returns:
    List of update ops of the layer that depend on `inputs`.

<hr>

*get_weights()*

Returns the current weights of the layer.

    The weights of a layer represent the state of the layer. This function
    returns both trainable and non-trainable weight values associated with this
    layer as a list of Numpy arrays, which can in turn be used to load state
    into similarly parameterized layers.

    For example, a Dense layer returns a list of two values-- per-output
    weights and the bias value. These can be used to set the weights of another
    Dense layer:

    ```
    >>> a = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(1.))
    >>> a_out = a(tf.convert_to_tensor([[1., 2., 3.]]))
    >>> a.get_weights()
    [array([[1.],
    [1.],
    [1.]], dtype=float32), array([0.], dtype=float32)]
    >>> b = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(2.))
    >>> b_out = b(tf.convert_to_tensor([[10., 20., 30.]]))
    >>> b.get_weights()
    [array([[2.],
    [2.],
    [2.]], dtype=float32), array([0.], dtype=float32)]
    >>> b.set_weights(a.get_weights())
    >>> b.get_weights()
    [array([[1.],
    [1.],
    [1.]], dtype=float32), array([0.], dtype=float32)]

    Returns:
    Weights values as a list of numpy arrays.
```

<hr>

*reset_state()*

Resets all of the metric state variables at the start of each epoch.

<hr>

*reset_states()*

Resets all of the metric state variables.

    This function is called between epochs/steps,
    when a metric is evaluated during training.

<hr>

*result()*

Computes and returns the metric value tensor.

    Result computation is an idempotent operation that simply calculates the
    metric value using the state variables.

<hr>

*set_weights(weights)*

Sets the weights of the layer, from Numpy arrays.

    The weights of a layer represent the state of the layer. This function
    sets the weight values from numpy arrays. The weight values should be
    passed in the order they are created by the layer. Note that the layer's
    weights must be instantiated before calling this function by calling
    the layer.

    For example, a Dense layer returns a list of two values-- per-output
    weights and the bias value. These can be used to set the weights of another
    Dense layer:

    ```
    >>> a = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(1.))
    >>> a_out = a(tf.convert_to_tensor([[1., 2., 3.]]))
    >>> a.get_weights()
    [array([[1.],
    [1.],
    [1.]], dtype=float32), array([0.], dtype=float32)]
    >>> b = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(2.))
    >>> b_out = b(tf.convert_to_tensor([[10., 20., 30.]]))
    >>> b.get_weights()
    [array([[2.],
    [2.],
    [2.]], dtype=float32), array([0.], dtype=float32)]
    >>> b.set_weights(a.get_weights())
    >>> b.get_weights()
    [array([[1.],
    [1.],
    [1.]], dtype=float32), array([0.], dtype=float32)]

    Arguments:
    weights: a list of Numpy arrays. The number
    of arrays and their shape must match
    number of the dimensions of the weights
    of the layer (i.e. it should match the
    output of `get_weights`).

    Raises:
    ValueError: If the provided weights list does not match the
    layer's specifications.
```

<hr>

*update_state(y_true, y_pred, sample_weight=None)*

Computes mean absolute error for ordinal labels.

    Args:
    y_true: Cumulatiuve logits from CondorOrdinal layer.
    y_pred: CondorOrdinal Encoded Labels.
    sample_weight (optional): Not implemented.

<hr>

*with_name_scope(method)*

Decorator to automatically enter the module name scope.

    ```
    >>> class MyModule(tf.Module):
    ...   @tf.Module.with_name_scope
    ...   def __call__(self, x):
    ...     if not hasattr(self, 'w'):
    ...       self.w = tf.Variable(tf.random.normal([x.shape[1], 3]))
    ...     return tf.matmul(x, self.w)

    Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
    names included the module name:

    >>> mod = MyModule()
    >>> mod(tf.ones([1, 2]))
    <tf.Tensor: shape=(1, 3), dtype=float32, numpy=..., dtype=float32)>
    >>> mod.w
    <tf.Variable 'my_module/Variable:0' shape=(2, 3) dtype=float32,
    numpy=..., dtype=float32)>

    Args:
    method: The method to wrap.

    Returns:
    The original method wrapped such that it enters the module's name scope.
```

### Properties

<hr>

*activity_regularizer*

Optional regularizer function for the output of this layer.

<hr>

*compute_dtype*

The dtype of the layer's computations.

    This is equivalent to `Layer.dtype_policy.compute_dtype`. Unless
    mixed precision is used, this is the same as `Layer.dtype`, the dtype of
    the weights.

    Layers automatically cast their inputs to the compute dtype, which causes
    computations and the output to be in the compute dtype as well. This is done
    by the base Layer class in `Layer.__call__`, so you do not have to insert
    these casts if implementing your own layer.

    Layers often perform certain internal computations in higher precision when
    `compute_dtype` is float16 or bfloat16 for numeric stability. The output
    will still typically be float16 or bfloat16 in such cases.

    Returns:
    The layer's compute dtype.

<hr>

*dtype*

The dtype of the layer weights.

    This is equivalent to `Layer.dtype_policy.variable_dtype`. Unless
    mixed precision is used, this is the same as `Layer.compute_dtype`, the
    dtype of the layer's computations.

<hr>

*dtype_policy*

The dtype policy associated with this layer.

    This is an instance of a `tf.keras.mixed_precision.Policy`.

<hr>

*dynamic*

Whether the layer is dynamic (eager-only); set in the constructor.

<hr>

*inbound_nodes*

Deprecated, do NOT use! Only for compatibility with external Keras.

<hr>

*input*

Retrieves the input tensor(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer.

    Returns:
    Input tensor or list of input tensors.

    Raises:
    RuntimeError: If called in Eager mode.
    AttributeError: If no inbound nodes are found.

<hr>

*input_mask*

Retrieves the input mask tensor(s) of a layer.

    Only applicable if the layer has exactly one inbound node,
    i.e. if it is connected to one incoming layer.

    Returns:
    Input mask tensor (potentially None) or list of input
    mask tensors.

    Raises:
    AttributeError: if the layer is connected to
    more than one incoming layers.

<hr>

*input_shape*

Retrieves the input shape(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer, or if all inputs
    have the same shape.

    Returns:
    Input shape, as an integer shape tuple
    (or list of shape tuples, one tuple per input tensor).

    Raises:
    AttributeError: if the layer has no defined input_shape.
    RuntimeError: if called in Eager mode.

<hr>

*input_spec*

`InputSpec` instance(s) describing the input format for this layer.

    When you create a layer subclass, you can set `self.input_spec` to enable
    the layer to run input compatibility checks when it is called.
    Consider a `Conv2D` layer: it can only be called on a single input tensor
    of rank 4. As such, you can set, in `__init__()`:

    ```python
    self.input_spec = tf.keras.layers.InputSpec(ndim=4)
    ```

    Now, if you try to call the layer on an input that isn't rank 4
    (for instance, an input of shape `(2,)`, it will raise a nicely-formatted
    error:

    ```
    ValueError: Input 0 of layer conv2d is incompatible with the layer:
    expected ndim=4, found ndim=1. Full shape received: [2]
    ```

    Input checks that can be specified via `input_spec` include:
    - Structure (e.g. a single input, a list of 2 inputs, etc)
    - Shape
    - Rank (ndim)
    - Dtype

    For more information, see `tf.keras.layers.InputSpec`.

    Returns:
    A `tf.keras.layers.InputSpec` instance, or nested structure thereof.

<hr>

*losses*

List of losses added using the `add_loss()` API.

    Variable regularization tensors are created when this property is accessed,
    so it is eager safe: accessing `losses` under a `tf.GradientTape` will
    propagate gradients back to the corresponding variables.

    Examples:

    ```
    >>> class MyLayer(tf.keras.layers.Layer):
    ...   def call(self, inputs):
    ...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
    ...     return inputs
    >>> l = MyLayer()
    >>> l(np.ones((10, 1)))
    >>> l.losses
    [1.0]

    >>> inputs = tf.keras.Input(shape=(10,))
    >>> x = tf.keras.layers.Dense(10)(inputs)
    >>> outputs = tf.keras.layers.Dense(1)(x)
    >>> model = tf.keras.Model(inputs, outputs)
    >>> # Activity regularization.
    >>> len(model.losses)
    0
    >>> model.add_loss(tf.abs(tf.reduce_mean(x)))
    >>> len(model.losses)
    1

    >>> inputs = tf.keras.Input(shape=(10,))
    >>> d = tf.keras.layers.Dense(10, kernel_initializer='ones')
    >>> x = d(inputs)
    >>> outputs = tf.keras.layers.Dense(1)(x)
    >>> model = tf.keras.Model(inputs, outputs)
    >>> # Weight regularization.
    >>> model.add_loss(lambda: tf.reduce_mean(d.kernel))
    >>> model.losses
    [<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]

    Returns:
    A list of tensors.
```

<hr>

*metrics*

List of metrics added using the `add_metric()` API.

    Example:

    ```
    >>> input = tf.keras.layers.Input(shape=(3,))
    >>> d = tf.keras.layers.Dense(2)
    >>> output = d(input)
    >>> d.add_metric(tf.reduce_max(output), name='max')
    >>> d.add_metric(tf.reduce_min(output), name='min')
    >>> [m.name for m in d.metrics]
    ['max', 'min']

    Returns:
    A list of `Metric` objects.
```

<hr>

*name*

Name of the layer (string), set in the constructor.

<hr>

*name_scope*

Returns a `tf.name_scope` instance for this class.

<hr>

*non_trainable_variables*

None

<hr>

*non_trainable_weights*

List of all non-trainable weights tracked by this layer.

    Non-trainable weights are *not* updated during training. They are expected
    to be updated manually in `call()`.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
    A list of non-trainable variables.

<hr>

*outbound_nodes*

Deprecated, do NOT use! Only for compatibility with external Keras.

<hr>

*output*

Retrieves the output tensor(s) of a layer.

    Only applicable if the layer has exactly one output,
    i.e. if it is connected to one incoming layer.

    Returns:
    Output tensor or list of output tensors.

    Raises:
    AttributeError: if the layer is connected to more than one incoming
    layers.
    RuntimeError: if called in Eager mode.

<hr>

*output_mask*

Retrieves the output mask tensor(s) of a layer.

    Only applicable if the layer has exactly one inbound node,
    i.e. if it is connected to one incoming layer.

    Returns:
    Output mask tensor (potentially None) or list of output
    mask tensors.

    Raises:
    AttributeError: if the layer is connected to
    more than one incoming layers.

<hr>

*output_shape*

Retrieves the output shape(s) of a layer.

    Only applicable if the layer has one output,
    or if all outputs have the same shape.

    Returns:
    Output shape, as an integer shape tuple
    (or list of shape tuples, one tuple per output tensor).

    Raises:
    AttributeError: if the layer has no defined output shape.
    RuntimeError: if called in Eager mode.

<hr>

*stateful*

None

<hr>

*submodules*

Sequence of all sub-modules.

    Submodules are modules which are properties of this module, or found as
    properties of modules which are properties of this module (and so on).

    ```
    >>> a = tf.Module()
    >>> b = tf.Module()
    >>> c = tf.Module()
    >>> a.b = b
    >>> b.c = c
    >>> list(a.submodules) == [b, c]
    True
    >>> list(b.submodules) == [c]
    True
    >>> list(c.submodules) == []
    True

    Returns:
    A sequence of all submodules.
```

<hr>

*supports_masking*

Whether this layer supports computing a mask using `compute_mask`.

<hr>

*trainable*

None

<hr>

*trainable_variables*

Sequence of trainable variables owned by this module and its submodules.

    Note: this method uses reflection to find variables on the current instance
    and submodules. For performance reasons you may wish to cache the result
    of calling this method if you don't expect the return value to change.

    Returns:
    A sequence of variables for the current module (sorted by attribute
    name) followed by variables from all submodules recursively (breadth
    first).

<hr>

*trainable_weights*

List of all trainable weights tracked by this layer.

    Trainable weights are updated via gradient descent during training.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
    A list of trainable variables.

<hr>

*updates*

None

<hr>

*variable_dtype*

Alias of `Layer.dtype`, the dtype of the weights.

<hr>

*variables*

Returns the list of all layer variables/weights.

    Alias of `self.weights`.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
    A list of variables.

<hr>

*weights*

Returns the list of all layer variables/weights.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
    A list of variables.

## SparseOrdinalMeanAbsoluteError

*SparseOrdinalMeanAbsoluteError(*args, **kwargs)*

Computes mean absolute error for ordinal labels.

### Methods

<hr>

*add_loss(losses, **kwargs)*

Add loss tensor(s), potentially dependent on layer inputs.

    Some losses (for instance, activity regularization losses) may be dependent
    on the inputs passed when calling a layer. Hence, when reusing the same
    layer on different inputs `a` and `b`, some entries in `layer.losses` may
    be dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    This method can be used inside a subclassed layer or model's `call`
    function, in which case `losses` should be a Tensor or list of Tensors.

    Example:

    ```python
    class MyLayer(tf.keras.layers.Layer):
    def call(self, inputs):
    self.add_loss(tf.abs(tf.reduce_mean(inputs)))
    return inputs
    ```

    This method can also be called directly on a Functional Model during
    construction. In this case, any loss Tensors passed to this Model must
    be symbolic and be able to be traced back to the model's `Input`s. These
    losses become part of the model's topology and are tracked in `get_config`.

    Example:

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    # Activity regularization.
    model.add_loss(tf.abs(tf.reduce_mean(x)))
    ```

    If this is not the case for your loss (if, for example, your loss references
    a `Variable` of one of the model's layers), you can wrap your loss in a
    zero-argument lambda. These losses are not tracked as part of the model's
    topology since they can't be serialized.

    Example:

    ```python
    inputs = tf.keras.Input(shape=(10,))
    d = tf.keras.layers.Dense(10)
    x = d(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    # Weight regularization.
    model.add_loss(lambda: tf.reduce_mean(d.kernel))
    ```

    Arguments:
    losses: Loss tensor, or list/tuple of tensors. Rather than tensors, losses
    may also be zero-argument callables which create a loss tensor.
**kwargs: Additional keyword arguments for backward compatibility.
    Accepted values:

inputs - Deprecated, will be automatically inferred.

<hr>

*add_metric(value, name=None, **kwargs)*

Adds metric tensor to the layer.

    This method can be used inside the `call()` method of a subclassed layer
    or model.

    ```python
    class MyMetricLayer(tf.keras.layers.Layer):
    def __init__(self):
    super(MyMetricLayer, self).__init__(name='my_metric_layer')
    self.mean = tf.keras.metrics.Mean(name='metric_1')

    def call(self, inputs):
    self.add_metric(self.mean(x))
    self.add_metric(tf.reduce_sum(x), name='metric_2')
    return inputs
    ```

    This method can also be called directly on a Functional Model during
    construction. In this case, any tensor passed to this Model must
    be symbolic and be able to be traced back to the model's `Input`s. These
    metrics become part of the model's topology and are tracked when you
    save the model via `save()`.

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.add_metric(math_ops.reduce_sum(x), name='metric_1')
    ```

    Note: Calling `add_metric()` with the result of a metric object on a
    Functional Model, as shown in the example below, is not supported. This is
    because we cannot trace the metric result tensor back to the model's inputs.

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.add_metric(tf.keras.metrics.Mean()(x), name='metric_1')
    ```

    Args:
    value: Metric tensor.
    name: String metric name.
**kwargs: Additional keyword arguments for backward compatibility.
    Accepted values:

`aggregation` - When the `value` tensor provided is not the result of
    calling a `keras.Metric` instance, it will be aggregated by default
    using a `keras.Metric.Mean`.

<hr>

*add_update(updates, inputs=None)*

Add update op(s), potentially dependent on layer inputs.

    Weight updates (for instance, the updates of the moving mean and variance
    in a BatchNormalization layer) may be dependent on the inputs passed
    when calling a layer. Hence, when reusing the same layer on
    different inputs `a` and `b`, some entries in `layer.updates` may be
    dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    This call is ignored when eager execution is enabled (in that case, variable
    updates are run on the fly and thus do not need to be tracked for later
    execution).

    Arguments:
    updates: Update op, or list/tuple of update ops, or zero-arg callable
    that returns an update op. A zero-arg callable should be passed in
    order to disable running the updates by setting `trainable=False`
    on this Layer, when executing in Eager mode.
    inputs: Deprecated, will be automatically inferred.

<hr>

*add_variable(*args, **kwargs)*

Deprecated, do NOT use! Alias for `add_weight`.

<hr>

*add_weight(name, shape=(), aggregation=<VariableAggregation.SUM: 1>, synchronization=<VariableSynchronization.ON_READ: 3>, initializer=None, dtype=None)*

Adds state variable. Only for use by subclasses.

<hr>

*apply(inputs, *args, **kwargs)*

Deprecated, do NOT use!

    This is an alias of `self.__call__`.

    Arguments:
    inputs: Input tensor(s).
    *args: additional positional arguments to be passed to `self.call`.
**kwargs: additional keyword arguments to be passed to `self.call`.


Returns:
    Output tensor(s).

<hr>

*build(input_shape)*

Creates the variables of the layer (optional, for subclass implementers).

    This is a method that implementers of subclasses of `Layer` or `Model`
    can override if they need a state-creation step in-between
    layer instantiation and layer call.

    This is typically used to create the weights of `Layer` subclasses.

    Arguments:
    input_shape: Instance of `TensorShape`, or list of instances of
    `TensorShape` if the layer expects a list of inputs
    (one instance per input).

<hr>

*call(inputs, **kwargs)*

This is where the layer's logic lives.

    Note here that `call()` method in `tf.keras` is little bit different
    from `keras` API. In `keras` API, you can pass support masking for
    layers as additional arguments. Whereas `tf.keras` has `compute_mask()`
    method to support masking.

    Arguments:
    inputs: Input tensor, or list/tuple of input tensors.
**kwargs: Additional keyword arguments. Currently unused.


Returns:
    A tensor or list/tuple of tensors.

<hr>

*compute_mask(inputs, mask=None)*

Computes an output mask tensor.

    Arguments:
    inputs: Tensor or list of tensors.
    mask: Tensor or list of tensors.

    Returns:
    None or a tensor (or list of tensors,
    one per output tensor of the layer).

<hr>

*compute_output_shape(input_shape)*

Computes the output shape of the layer.

    If the layer has not been built, this method will call `build` on the
    layer. This assumes that the layer will later be used with inputs that
    match the input shape provided here.

    Arguments:
    input_shape: Shape tuple (tuple of integers)
    or list of shape tuples (one per output tensor of the layer).
    Shape tuples can include None for free dimensions,
    instead of an integer.

    Returns:
    An input shape tuple.

<hr>

*compute_output_signature(input_signature)*

Compute the output tensor signature of the layer based on the inputs.

    Unlike a TensorShape object, a TensorSpec object contains both shape
    and dtype information for a tensor. This method allows layers to provide
    output dtype information if it is different from the input dtype.
    For any layer that doesn't implement this function,
    the framework will fall back to use `compute_output_shape`, and will
    assume that the output dtype matches the input dtype.

    Args:
    input_signature: Single TensorSpec or nested structure of TensorSpec
    objects, describing a candidate input for the layer.

    Returns:
    Single TensorSpec or nested structure of TensorSpec objects, describing
    how the layer would transform the provided input.

    Raises:
    TypeError: If input_signature contains a non-TensorSpec object.

<hr>

*count_params()*

Count the total number of scalars composing the weights.

    Returns:
    An integer count.

    Raises:
    ValueError: if the layer isn't yet built
    (in which case its weights aren't yet defined).

<hr>

*from_config(config)*

Creates a layer from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same layer from the config
    dictionary. It does not handle layer connectivity
    (handled by Network), nor weights (handled by `set_weights`).

    Arguments:
    config: A Python dictionary, typically the
    output of get_config.

    Returns:
    A layer instance.

<hr>

*get_config()*

Returns the serializable config of the metric.

<hr>

*get_input_at(node_index)*

Retrieves the input tensor(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A tensor (or list of tensors if the layer has multiple inputs).

    Raises:
    RuntimeError: If called in Eager mode.

<hr>

*get_input_mask_at(node_index)*

Retrieves the input mask tensor(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A mask tensor
    (or list of tensors if the layer has multiple inputs).

<hr>

*get_input_shape_at(node_index)*

Retrieves the input shape(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A shape tuple
    (or list of shape tuples if the layer has multiple inputs).

    Raises:
    RuntimeError: If called in Eager mode.

<hr>

*get_losses_for(inputs)*

Deprecated, do NOT use!

    Retrieves losses relevant to a specific set of inputs.

    Arguments:
    inputs: Input tensor or list/tuple of input tensors.

    Returns:
    List of loss tensors of the layer that depend on `inputs`.

<hr>

*get_output_at(node_index)*

Retrieves the output tensor(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A tensor (or list of tensors if the layer has multiple outputs).

    Raises:
    RuntimeError: If called in Eager mode.

<hr>

*get_output_mask_at(node_index)*

Retrieves the output mask tensor(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A mask tensor
    (or list of tensors if the layer has multiple outputs).

<hr>

*get_output_shape_at(node_index)*

Retrieves the output shape(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A shape tuple
    (or list of shape tuples if the layer has multiple outputs).

    Raises:
    RuntimeError: If called in Eager mode.

<hr>

*get_updates_for(inputs)*

Deprecated, do NOT use!

    Retrieves updates relevant to a specific set of inputs.

    Arguments:
    inputs: Input tensor or list/tuple of input tensors.

    Returns:
    List of update ops of the layer that depend on `inputs`.

<hr>

*get_weights()*

Returns the current weights of the layer.

    The weights of a layer represent the state of the layer. This function
    returns both trainable and non-trainable weight values associated with this
    layer as a list of Numpy arrays, which can in turn be used to load state
    into similarly parameterized layers.

    For example, a Dense layer returns a list of two values-- per-output
    weights and the bias value. These can be used to set the weights of another
    Dense layer:

    ```
    >>> a = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(1.))
    >>> a_out = a(tf.convert_to_tensor([[1., 2., 3.]]))
    >>> a.get_weights()
    [array([[1.],
    [1.],
    [1.]], dtype=float32), array([0.], dtype=float32)]
    >>> b = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(2.))
    >>> b_out = b(tf.convert_to_tensor([[10., 20., 30.]]))
    >>> b.get_weights()
    [array([[2.],
    [2.],
    [2.]], dtype=float32), array([0.], dtype=float32)]
    >>> b.set_weights(a.get_weights())
    >>> b.get_weights()
    [array([[1.],
    [1.],
    [1.]], dtype=float32), array([0.], dtype=float32)]

    Returns:
    Weights values as a list of numpy arrays.
```

<hr>

*reset_state()*

Resets all of the metric state variables at the start of each epoch.

<hr>

*reset_states()*

Resets all of the metric state variables.

    This function is called between epochs/steps,
    when a metric is evaluated during training.

<hr>

*result()*

Computes and returns the metric value tensor.

    Result computation is an idempotent operation that simply calculates the
    metric value using the state variables.

<hr>

*set_weights(weights)*

Sets the weights of the layer, from Numpy arrays.

    The weights of a layer represent the state of the layer. This function
    sets the weight values from numpy arrays. The weight values should be
    passed in the order they are created by the layer. Note that the layer's
    weights must be instantiated before calling this function by calling
    the layer.

    For example, a Dense layer returns a list of two values-- per-output
    weights and the bias value. These can be used to set the weights of another
    Dense layer:

    ```
    >>> a = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(1.))
    >>> a_out = a(tf.convert_to_tensor([[1., 2., 3.]]))
    >>> a.get_weights()
    [array([[1.],
    [1.],
    [1.]], dtype=float32), array([0.], dtype=float32)]
    >>> b = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(2.))
    >>> b_out = b(tf.convert_to_tensor([[10., 20., 30.]]))
    >>> b.get_weights()
    [array([[2.],
    [2.],
    [2.]], dtype=float32), array([0.], dtype=float32)]
    >>> b.set_weights(a.get_weights())
    >>> b.get_weights()
    [array([[1.],
    [1.],
    [1.]], dtype=float32), array([0.], dtype=float32)]

    Arguments:
    weights: a list of Numpy arrays. The number
    of arrays and their shape must match
    number of the dimensions of the weights
    of the layer (i.e. it should match the
    output of `get_weights`).

    Raises:
    ValueError: If the provided weights list does not match the
    layer's specifications.
```

<hr>

*update_state(y_true, y_pred, sample_weight=None)*

Computes mean absolute error for ordinal labels.

    Args:
    y_true: Cumulatiuve logits from CondorOrdinal layer.
    y_pred: CondorOrdinal Encoded Labels.
    sample_weight (optional): Not implemented.

<hr>

*with_name_scope(method)*

Decorator to automatically enter the module name scope.

    ```
    >>> class MyModule(tf.Module):
    ...   @tf.Module.with_name_scope
    ...   def __call__(self, x):
    ...     if not hasattr(self, 'w'):
    ...       self.w = tf.Variable(tf.random.normal([x.shape[1], 3]))
    ...     return tf.matmul(x, self.w)

    Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
    names included the module name:

    >>> mod = MyModule()
    >>> mod(tf.ones([1, 2]))
    <tf.Tensor: shape=(1, 3), dtype=float32, numpy=..., dtype=float32)>
    >>> mod.w
    <tf.Variable 'my_module/Variable:0' shape=(2, 3) dtype=float32,
    numpy=..., dtype=float32)>

    Args:
    method: The method to wrap.

    Returns:
    The original method wrapped such that it enters the module's name scope.
```

### Properties

<hr>

*activity_regularizer*

Optional regularizer function for the output of this layer.

<hr>

*compute_dtype*

The dtype of the layer's computations.

    This is equivalent to `Layer.dtype_policy.compute_dtype`. Unless
    mixed precision is used, this is the same as `Layer.dtype`, the dtype of
    the weights.

    Layers automatically cast their inputs to the compute dtype, which causes
    computations and the output to be in the compute dtype as well. This is done
    by the base Layer class in `Layer.__call__`, so you do not have to insert
    these casts if implementing your own layer.

    Layers often perform certain internal computations in higher precision when
    `compute_dtype` is float16 or bfloat16 for numeric stability. The output
    will still typically be float16 or bfloat16 in such cases.

    Returns:
    The layer's compute dtype.

<hr>

*dtype*

The dtype of the layer weights.

    This is equivalent to `Layer.dtype_policy.variable_dtype`. Unless
    mixed precision is used, this is the same as `Layer.compute_dtype`, the
    dtype of the layer's computations.

<hr>

*dtype_policy*

The dtype policy associated with this layer.

    This is an instance of a `tf.keras.mixed_precision.Policy`.

<hr>

*dynamic*

Whether the layer is dynamic (eager-only); set in the constructor.

<hr>

*inbound_nodes*

Deprecated, do NOT use! Only for compatibility with external Keras.

<hr>

*input*

Retrieves the input tensor(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer.

    Returns:
    Input tensor or list of input tensors.

    Raises:
    RuntimeError: If called in Eager mode.
    AttributeError: If no inbound nodes are found.

<hr>

*input_mask*

Retrieves the input mask tensor(s) of a layer.

    Only applicable if the layer has exactly one inbound node,
    i.e. if it is connected to one incoming layer.

    Returns:
    Input mask tensor (potentially None) or list of input
    mask tensors.

    Raises:
    AttributeError: if the layer is connected to
    more than one incoming layers.

<hr>

*input_shape*

Retrieves the input shape(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer, or if all inputs
    have the same shape.

    Returns:
    Input shape, as an integer shape tuple
    (or list of shape tuples, one tuple per input tensor).

    Raises:
    AttributeError: if the layer has no defined input_shape.
    RuntimeError: if called in Eager mode.

<hr>

*input_spec*

`InputSpec` instance(s) describing the input format for this layer.

    When you create a layer subclass, you can set `self.input_spec` to enable
    the layer to run input compatibility checks when it is called.
    Consider a `Conv2D` layer: it can only be called on a single input tensor
    of rank 4. As such, you can set, in `__init__()`:

    ```python
    self.input_spec = tf.keras.layers.InputSpec(ndim=4)
    ```

    Now, if you try to call the layer on an input that isn't rank 4
    (for instance, an input of shape `(2,)`, it will raise a nicely-formatted
    error:

    ```
    ValueError: Input 0 of layer conv2d is incompatible with the layer:
    expected ndim=4, found ndim=1. Full shape received: [2]
    ```

    Input checks that can be specified via `input_spec` include:
    - Structure (e.g. a single input, a list of 2 inputs, etc)
    - Shape
    - Rank (ndim)
    - Dtype

    For more information, see `tf.keras.layers.InputSpec`.

    Returns:
    A `tf.keras.layers.InputSpec` instance, or nested structure thereof.

<hr>

*losses*

List of losses added using the `add_loss()` API.

    Variable regularization tensors are created when this property is accessed,
    so it is eager safe: accessing `losses` under a `tf.GradientTape` will
    propagate gradients back to the corresponding variables.

    Examples:

    ```
    >>> class MyLayer(tf.keras.layers.Layer):
    ...   def call(self, inputs):
    ...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
    ...     return inputs
    >>> l = MyLayer()
    >>> l(np.ones((10, 1)))
    >>> l.losses
    [1.0]

    >>> inputs = tf.keras.Input(shape=(10,))
    >>> x = tf.keras.layers.Dense(10)(inputs)
    >>> outputs = tf.keras.layers.Dense(1)(x)
    >>> model = tf.keras.Model(inputs, outputs)
    >>> # Activity regularization.
    >>> len(model.losses)
    0
    >>> model.add_loss(tf.abs(tf.reduce_mean(x)))
    >>> len(model.losses)
    1

    >>> inputs = tf.keras.Input(shape=(10,))
    >>> d = tf.keras.layers.Dense(10, kernel_initializer='ones')
    >>> x = d(inputs)
    >>> outputs = tf.keras.layers.Dense(1)(x)
    >>> model = tf.keras.Model(inputs, outputs)
    >>> # Weight regularization.
    >>> model.add_loss(lambda: tf.reduce_mean(d.kernel))
    >>> model.losses
    [<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]

    Returns:
    A list of tensors.
```

<hr>

*metrics*

List of metrics added using the `add_metric()` API.

    Example:

    ```
    >>> input = tf.keras.layers.Input(shape=(3,))
    >>> d = tf.keras.layers.Dense(2)
    >>> output = d(input)
    >>> d.add_metric(tf.reduce_max(output), name='max')
    >>> d.add_metric(tf.reduce_min(output), name='min')
    >>> [m.name for m in d.metrics]
    ['max', 'min']

    Returns:
    A list of `Metric` objects.
```

<hr>

*name*

Name of the layer (string), set in the constructor.

<hr>

*name_scope*

Returns a `tf.name_scope` instance for this class.

<hr>

*non_trainable_variables*

None

<hr>

*non_trainable_weights*

List of all non-trainable weights tracked by this layer.

    Non-trainable weights are *not* updated during training. They are expected
    to be updated manually in `call()`.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
    A list of non-trainable variables.

<hr>

*outbound_nodes*

Deprecated, do NOT use! Only for compatibility with external Keras.

<hr>

*output*

Retrieves the output tensor(s) of a layer.

    Only applicable if the layer has exactly one output,
    i.e. if it is connected to one incoming layer.

    Returns:
    Output tensor or list of output tensors.

    Raises:
    AttributeError: if the layer is connected to more than one incoming
    layers.
    RuntimeError: if called in Eager mode.

<hr>

*output_mask*

Retrieves the output mask tensor(s) of a layer.

    Only applicable if the layer has exactly one inbound node,
    i.e. if it is connected to one incoming layer.

    Returns:
    Output mask tensor (potentially None) or list of output
    mask tensors.

    Raises:
    AttributeError: if the layer is connected to
    more than one incoming layers.

<hr>

*output_shape*

Retrieves the output shape(s) of a layer.

    Only applicable if the layer has one output,
    or if all outputs have the same shape.

    Returns:
    Output shape, as an integer shape tuple
    (or list of shape tuples, one tuple per output tensor).

    Raises:
    AttributeError: if the layer has no defined output shape.
    RuntimeError: if called in Eager mode.

<hr>

*stateful*

None

<hr>

*submodules*

Sequence of all sub-modules.

    Submodules are modules which are properties of this module, or found as
    properties of modules which are properties of this module (and so on).

    ```
    >>> a = tf.Module()
    >>> b = tf.Module()
    >>> c = tf.Module()
    >>> a.b = b
    >>> b.c = c
    >>> list(a.submodules) == [b, c]
    True
    >>> list(b.submodules) == [c]
    True
    >>> list(c.submodules) == []
    True

    Returns:
    A sequence of all submodules.
```

<hr>

*supports_masking*

Whether this layer supports computing a mask using `compute_mask`.

<hr>

*trainable*

None

<hr>

*trainable_variables*

Sequence of trainable variables owned by this module and its submodules.

    Note: this method uses reflection to find variables on the current instance
    and submodules. For performance reasons you may wish to cache the result
    of calling this method if you don't expect the return value to change.

    Returns:
    A sequence of variables for the current module (sorted by attribute
    name) followed by variables from all submodules recursively (breadth
    first).

<hr>

*trainable_weights*

List of all trainable weights tracked by this layer.

    Trainable weights are updated via gradient descent during training.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
    A list of trainable variables.

<hr>

*updates*

None

<hr>

*variable_dtype*

Alias of `Layer.dtype`, the dtype of the weights.

<hr>

*variables*

Returns the list of all layer variables/weights.

    Alias of `self.weights`.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
    A list of variables.

<hr>

*weights*

Returns the list of all layer variables/weights.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
    A list of variables.

## SparseOrdinalEarthMoversDistance

*SparseOrdinalEarthMoversDistance(*args, **kwargs)*

Computes earth movers distance for ordinal labels.

### Methods

<hr>

*add_loss(losses, **kwargs)*

Add loss tensor(s), potentially dependent on layer inputs.

    Some losses (for instance, activity regularization losses) may be dependent
    on the inputs passed when calling a layer. Hence, when reusing the same
    layer on different inputs `a` and `b`, some entries in `layer.losses` may
    be dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    This method can be used inside a subclassed layer or model's `call`
    function, in which case `losses` should be a Tensor or list of Tensors.

    Example:

    ```python
    class MyLayer(tf.keras.layers.Layer):
    def call(self, inputs):
    self.add_loss(tf.abs(tf.reduce_mean(inputs)))
    return inputs
    ```

    This method can also be called directly on a Functional Model during
    construction. In this case, any loss Tensors passed to this Model must
    be symbolic and be able to be traced back to the model's `Input`s. These
    losses become part of the model's topology and are tracked in `get_config`.

    Example:

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    # Activity regularization.
    model.add_loss(tf.abs(tf.reduce_mean(x)))
    ```

    If this is not the case for your loss (if, for example, your loss references
    a `Variable` of one of the model's layers), you can wrap your loss in a
    zero-argument lambda. These losses are not tracked as part of the model's
    topology since they can't be serialized.

    Example:

    ```python
    inputs = tf.keras.Input(shape=(10,))
    d = tf.keras.layers.Dense(10)
    x = d(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    # Weight regularization.
    model.add_loss(lambda: tf.reduce_mean(d.kernel))
    ```

    Arguments:
    losses: Loss tensor, or list/tuple of tensors. Rather than tensors, losses
    may also be zero-argument callables which create a loss tensor.
**kwargs: Additional keyword arguments for backward compatibility.
    Accepted values:

inputs - Deprecated, will be automatically inferred.

<hr>

*add_metric(value, name=None, **kwargs)*

Adds metric tensor to the layer.

    This method can be used inside the `call()` method of a subclassed layer
    or model.

    ```python
    class MyMetricLayer(tf.keras.layers.Layer):
    def __init__(self):
    super(MyMetricLayer, self).__init__(name='my_metric_layer')
    self.mean = tf.keras.metrics.Mean(name='metric_1')

    def call(self, inputs):
    self.add_metric(self.mean(x))
    self.add_metric(tf.reduce_sum(x), name='metric_2')
    return inputs
    ```

    This method can also be called directly on a Functional Model during
    construction. In this case, any tensor passed to this Model must
    be symbolic and be able to be traced back to the model's `Input`s. These
    metrics become part of the model's topology and are tracked when you
    save the model via `save()`.

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.add_metric(math_ops.reduce_sum(x), name='metric_1')
    ```

    Note: Calling `add_metric()` with the result of a metric object on a
    Functional Model, as shown in the example below, is not supported. This is
    because we cannot trace the metric result tensor back to the model's inputs.

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.add_metric(tf.keras.metrics.Mean()(x), name='metric_1')
    ```

    Args:
    value: Metric tensor.
    name: String metric name.
**kwargs: Additional keyword arguments for backward compatibility.
    Accepted values:

`aggregation` - When the `value` tensor provided is not the result of
    calling a `keras.Metric` instance, it will be aggregated by default
    using a `keras.Metric.Mean`.

<hr>

*add_update(updates, inputs=None)*

Add update op(s), potentially dependent on layer inputs.

    Weight updates (for instance, the updates of the moving mean and variance
    in a BatchNormalization layer) may be dependent on the inputs passed
    when calling a layer. Hence, when reusing the same layer on
    different inputs `a` and `b`, some entries in `layer.updates` may be
    dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    This call is ignored when eager execution is enabled (in that case, variable
    updates are run on the fly and thus do not need to be tracked for later
    execution).

    Arguments:
    updates: Update op, or list/tuple of update ops, or zero-arg callable
    that returns an update op. A zero-arg callable should be passed in
    order to disable running the updates by setting `trainable=False`
    on this Layer, when executing in Eager mode.
    inputs: Deprecated, will be automatically inferred.

<hr>

*add_variable(*args, **kwargs)*

Deprecated, do NOT use! Alias for `add_weight`.

<hr>

*add_weight(name, shape=(), aggregation=<VariableAggregation.SUM: 1>, synchronization=<VariableSynchronization.ON_READ: 3>, initializer=None, dtype=None)*

Adds state variable. Only for use by subclasses.

<hr>

*apply(inputs, *args, **kwargs)*

Deprecated, do NOT use!

    This is an alias of `self.__call__`.

    Arguments:
    inputs: Input tensor(s).
    *args: additional positional arguments to be passed to `self.call`.
**kwargs: additional keyword arguments to be passed to `self.call`.


Returns:
    Output tensor(s).

<hr>

*build(input_shape)*

Creates the variables of the layer (optional, for subclass implementers).

    This is a method that implementers of subclasses of `Layer` or `Model`
    can override if they need a state-creation step in-between
    layer instantiation and layer call.

    This is typically used to create the weights of `Layer` subclasses.

    Arguments:
    input_shape: Instance of `TensorShape`, or list of instances of
    `TensorShape` if the layer expects a list of inputs
    (one instance per input).

<hr>

*call(inputs, **kwargs)*

This is where the layer's logic lives.

    Note here that `call()` method in `tf.keras` is little bit different
    from `keras` API. In `keras` API, you can pass support masking for
    layers as additional arguments. Whereas `tf.keras` has `compute_mask()`
    method to support masking.

    Arguments:
    inputs: Input tensor, or list/tuple of input tensors.
**kwargs: Additional keyword arguments. Currently unused.


Returns:
    A tensor or list/tuple of tensors.

<hr>

*compute_mask(inputs, mask=None)*

Computes an output mask tensor.

    Arguments:
    inputs: Tensor or list of tensors.
    mask: Tensor or list of tensors.

    Returns:
    None or a tensor (or list of tensors,
    one per output tensor of the layer).

<hr>

*compute_output_shape(input_shape)*

Computes the output shape of the layer.

    If the layer has not been built, this method will call `build` on the
    layer. This assumes that the layer will later be used with inputs that
    match the input shape provided here.

    Arguments:
    input_shape: Shape tuple (tuple of integers)
    or list of shape tuples (one per output tensor of the layer).
    Shape tuples can include None for free dimensions,
    instead of an integer.

    Returns:
    An input shape tuple.

<hr>

*compute_output_signature(input_signature)*

Compute the output tensor signature of the layer based on the inputs.

    Unlike a TensorShape object, a TensorSpec object contains both shape
    and dtype information for a tensor. This method allows layers to provide
    output dtype information if it is different from the input dtype.
    For any layer that doesn't implement this function,
    the framework will fall back to use `compute_output_shape`, and will
    assume that the output dtype matches the input dtype.

    Args:
    input_signature: Single TensorSpec or nested structure of TensorSpec
    objects, describing a candidate input for the layer.

    Returns:
    Single TensorSpec or nested structure of TensorSpec objects, describing
    how the layer would transform the provided input.

    Raises:
    TypeError: If input_signature contains a non-TensorSpec object.

<hr>

*count_params()*

Count the total number of scalars composing the weights.

    Returns:
    An integer count.

    Raises:
    ValueError: if the layer isn't yet built
    (in which case its weights aren't yet defined).

<hr>

*from_config(config)*

Creates a layer from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same layer from the config
    dictionary. It does not handle layer connectivity
    (handled by Network), nor weights (handled by `set_weights`).

    Arguments:
    config: A Python dictionary, typically the
    output of get_config.

    Returns:
    A layer instance.

<hr>

*get_config()*

Returns the serializable config of the metric.

<hr>

*get_input_at(node_index)*

Retrieves the input tensor(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A tensor (or list of tensors if the layer has multiple inputs).

    Raises:
    RuntimeError: If called in Eager mode.

<hr>

*get_input_mask_at(node_index)*

Retrieves the input mask tensor(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A mask tensor
    (or list of tensors if the layer has multiple inputs).

<hr>

*get_input_shape_at(node_index)*

Retrieves the input shape(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A shape tuple
    (or list of shape tuples if the layer has multiple inputs).

    Raises:
    RuntimeError: If called in Eager mode.

<hr>

*get_losses_for(inputs)*

Deprecated, do NOT use!

    Retrieves losses relevant to a specific set of inputs.

    Arguments:
    inputs: Input tensor or list/tuple of input tensors.

    Returns:
    List of loss tensors of the layer that depend on `inputs`.

<hr>

*get_output_at(node_index)*

Retrieves the output tensor(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A tensor (or list of tensors if the layer has multiple outputs).

    Raises:
    RuntimeError: If called in Eager mode.

<hr>

*get_output_mask_at(node_index)*

Retrieves the output mask tensor(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A mask tensor
    (or list of tensors if the layer has multiple outputs).

<hr>

*get_output_shape_at(node_index)*

Retrieves the output shape(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A shape tuple
    (or list of shape tuples if the layer has multiple outputs).

    Raises:
    RuntimeError: If called in Eager mode.

<hr>

*get_updates_for(inputs)*

Deprecated, do NOT use!

    Retrieves updates relevant to a specific set of inputs.

    Arguments:
    inputs: Input tensor or list/tuple of input tensors.

    Returns:
    List of update ops of the layer that depend on `inputs`.

<hr>

*get_weights()*

Returns the current weights of the layer.

    The weights of a layer represent the state of the layer. This function
    returns both trainable and non-trainable weight values associated with this
    layer as a list of Numpy arrays, which can in turn be used to load state
    into similarly parameterized layers.

    For example, a Dense layer returns a list of two values-- per-output
    weights and the bias value. These can be used to set the weights of another
    Dense layer:

    ```
    >>> a = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(1.))
    >>> a_out = a(tf.convert_to_tensor([[1., 2., 3.]]))
    >>> a.get_weights()
    [array([[1.],
    [1.],
    [1.]], dtype=float32), array([0.], dtype=float32)]
    >>> b = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(2.))
    >>> b_out = b(tf.convert_to_tensor([[10., 20., 30.]]))
    >>> b.get_weights()
    [array([[2.],
    [2.],
    [2.]], dtype=float32), array([0.], dtype=float32)]
    >>> b.set_weights(a.get_weights())
    >>> b.get_weights()
    [array([[1.],
    [1.],
    [1.]], dtype=float32), array([0.], dtype=float32)]

    Returns:
    Weights values as a list of numpy arrays.
```

<hr>

*reset_state()*

Resets all of the metric state variables at the start of each epoch.

<hr>

*reset_states()*

Resets all of the metric state variables.

    This function is called between epochs/steps,
    when a metric is evaluated during training.

<hr>

*result()*

Computes and returns the metric value tensor.

    Result computation is an idempotent operation that simply calculates the
    metric value using the state variables.

<hr>

*set_weights(weights)*

Sets the weights of the layer, from Numpy arrays.

    The weights of a layer represent the state of the layer. This function
    sets the weight values from numpy arrays. The weight values should be
    passed in the order they are created by the layer. Note that the layer's
    weights must be instantiated before calling this function by calling
    the layer.

    For example, a Dense layer returns a list of two values-- per-output
    weights and the bias value. These can be used to set the weights of another
    Dense layer:

    ```
    >>> a = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(1.))
    >>> a_out = a(tf.convert_to_tensor([[1., 2., 3.]]))
    >>> a.get_weights()
    [array([[1.],
    [1.],
    [1.]], dtype=float32), array([0.], dtype=float32)]
    >>> b = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(2.))
    >>> b_out = b(tf.convert_to_tensor([[10., 20., 30.]]))
    >>> b.get_weights()
    [array([[2.],
    [2.],
    [2.]], dtype=float32), array([0.], dtype=float32)]
    >>> b.set_weights(a.get_weights())
    >>> b.get_weights()
    [array([[1.],
    [1.],
    [1.]], dtype=float32), array([0.], dtype=float32)]

    Arguments:
    weights: a list of Numpy arrays. The number
    of arrays and their shape must match
    number of the dimensions of the weights
    of the layer (i.e. it should match the
    output of `get_weights`).

    Raises:
    ValueError: If the provided weights list does not match the
    layer's specifications.
```

<hr>

*update_state(y_true, y_pred, sample_weight=None)*

Computes mean absolute error for ordinal labels.

    Args:
    y_true: Cumulatiuve logits from CondorOrdinal layer.
    y_pred: Sparse Labels with values in {0,1,...,num_classes-1}
    sample_weight (optional): Not implemented.

<hr>

*with_name_scope(method)*

Decorator to automatically enter the module name scope.

    ```
    >>> class MyModule(tf.Module):
    ...   @tf.Module.with_name_scope
    ...   def __call__(self, x):
    ...     if not hasattr(self, 'w'):
    ...       self.w = tf.Variable(tf.random.normal([x.shape[1], 3]))
    ...     return tf.matmul(x, self.w)

    Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
    names included the module name:

    >>> mod = MyModule()
    >>> mod(tf.ones([1, 2]))
    <tf.Tensor: shape=(1, 3), dtype=float32, numpy=..., dtype=float32)>
    >>> mod.w
    <tf.Variable 'my_module/Variable:0' shape=(2, 3) dtype=float32,
    numpy=..., dtype=float32)>

    Args:
    method: The method to wrap.

    Returns:
    The original method wrapped such that it enters the module's name scope.
```

### Properties

<hr>

*activity_regularizer*

Optional regularizer function for the output of this layer.

<hr>

*compute_dtype*

The dtype of the layer's computations.

    This is equivalent to `Layer.dtype_policy.compute_dtype`. Unless
    mixed precision is used, this is the same as `Layer.dtype`, the dtype of
    the weights.

    Layers automatically cast their inputs to the compute dtype, which causes
    computations and the output to be in the compute dtype as well. This is done
    by the base Layer class in `Layer.__call__`, so you do not have to insert
    these casts if implementing your own layer.

    Layers often perform certain internal computations in higher precision when
    `compute_dtype` is float16 or bfloat16 for numeric stability. The output
    will still typically be float16 or bfloat16 in such cases.

    Returns:
    The layer's compute dtype.

<hr>

*dtype*

The dtype of the layer weights.

    This is equivalent to `Layer.dtype_policy.variable_dtype`. Unless
    mixed precision is used, this is the same as `Layer.compute_dtype`, the
    dtype of the layer's computations.

<hr>

*dtype_policy*

The dtype policy associated with this layer.

    This is an instance of a `tf.keras.mixed_precision.Policy`.

<hr>

*dynamic*

Whether the layer is dynamic (eager-only); set in the constructor.

<hr>

*inbound_nodes*

Deprecated, do NOT use! Only for compatibility with external Keras.

<hr>

*input*

Retrieves the input tensor(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer.

    Returns:
    Input tensor or list of input tensors.

    Raises:
    RuntimeError: If called in Eager mode.
    AttributeError: If no inbound nodes are found.

<hr>

*input_mask*

Retrieves the input mask tensor(s) of a layer.

    Only applicable if the layer has exactly one inbound node,
    i.e. if it is connected to one incoming layer.

    Returns:
    Input mask tensor (potentially None) or list of input
    mask tensors.

    Raises:
    AttributeError: if the layer is connected to
    more than one incoming layers.

<hr>

*input_shape*

Retrieves the input shape(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer, or if all inputs
    have the same shape.

    Returns:
    Input shape, as an integer shape tuple
    (or list of shape tuples, one tuple per input tensor).

    Raises:
    AttributeError: if the layer has no defined input_shape.
    RuntimeError: if called in Eager mode.

<hr>

*input_spec*

`InputSpec` instance(s) describing the input format for this layer.

    When you create a layer subclass, you can set `self.input_spec` to enable
    the layer to run input compatibility checks when it is called.
    Consider a `Conv2D` layer: it can only be called on a single input tensor
    of rank 4. As such, you can set, in `__init__()`:

    ```python
    self.input_spec = tf.keras.layers.InputSpec(ndim=4)
    ```

    Now, if you try to call the layer on an input that isn't rank 4
    (for instance, an input of shape `(2,)`, it will raise a nicely-formatted
    error:

    ```
    ValueError: Input 0 of layer conv2d is incompatible with the layer:
    expected ndim=4, found ndim=1. Full shape received: [2]
    ```

    Input checks that can be specified via `input_spec` include:
    - Structure (e.g. a single input, a list of 2 inputs, etc)
    - Shape
    - Rank (ndim)
    - Dtype

    For more information, see `tf.keras.layers.InputSpec`.

    Returns:
    A `tf.keras.layers.InputSpec` instance, or nested structure thereof.

<hr>

*losses*

List of losses added using the `add_loss()` API.

    Variable regularization tensors are created when this property is accessed,
    so it is eager safe: accessing `losses` under a `tf.GradientTape` will
    propagate gradients back to the corresponding variables.

    Examples:

    ```
    >>> class MyLayer(tf.keras.layers.Layer):
    ...   def call(self, inputs):
    ...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
    ...     return inputs
    >>> l = MyLayer()
    >>> l(np.ones((10, 1)))
    >>> l.losses
    [1.0]

    >>> inputs = tf.keras.Input(shape=(10,))
    >>> x = tf.keras.layers.Dense(10)(inputs)
    >>> outputs = tf.keras.layers.Dense(1)(x)
    >>> model = tf.keras.Model(inputs, outputs)
    >>> # Activity regularization.
    >>> len(model.losses)
    0
    >>> model.add_loss(tf.abs(tf.reduce_mean(x)))
    >>> len(model.losses)
    1

    >>> inputs = tf.keras.Input(shape=(10,))
    >>> d = tf.keras.layers.Dense(10, kernel_initializer='ones')
    >>> x = d(inputs)
    >>> outputs = tf.keras.layers.Dense(1)(x)
    >>> model = tf.keras.Model(inputs, outputs)
    >>> # Weight regularization.
    >>> model.add_loss(lambda: tf.reduce_mean(d.kernel))
    >>> model.losses
    [<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]

    Returns:
    A list of tensors.
```

<hr>

*metrics*

List of metrics added using the `add_metric()` API.

    Example:

    ```
    >>> input = tf.keras.layers.Input(shape=(3,))
    >>> d = tf.keras.layers.Dense(2)
    >>> output = d(input)
    >>> d.add_metric(tf.reduce_max(output), name='max')
    >>> d.add_metric(tf.reduce_min(output), name='min')
    >>> [m.name for m in d.metrics]
    ['max', 'min']

    Returns:
    A list of `Metric` objects.
```

<hr>

*name*

Name of the layer (string), set in the constructor.

<hr>

*name_scope*

Returns a `tf.name_scope` instance for this class.

<hr>

*non_trainable_variables*

None

<hr>

*non_trainable_weights*

List of all non-trainable weights tracked by this layer.

    Non-trainable weights are *not* updated during training. They are expected
    to be updated manually in `call()`.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
    A list of non-trainable variables.

<hr>

*outbound_nodes*

Deprecated, do NOT use! Only for compatibility with external Keras.

<hr>

*output*

Retrieves the output tensor(s) of a layer.

    Only applicable if the layer has exactly one output,
    i.e. if it is connected to one incoming layer.

    Returns:
    Output tensor or list of output tensors.

    Raises:
    AttributeError: if the layer is connected to more than one incoming
    layers.
    RuntimeError: if called in Eager mode.

<hr>

*output_mask*

Retrieves the output mask tensor(s) of a layer.

    Only applicable if the layer has exactly one inbound node,
    i.e. if it is connected to one incoming layer.

    Returns:
    Output mask tensor (potentially None) or list of output
    mask tensors.

    Raises:
    AttributeError: if the layer is connected to
    more than one incoming layers.

<hr>

*output_shape*

Retrieves the output shape(s) of a layer.

    Only applicable if the layer has one output,
    or if all outputs have the same shape.

    Returns:
    Output shape, as an integer shape tuple
    (or list of shape tuples, one tuple per output tensor).

    Raises:
    AttributeError: if the layer has no defined output shape.
    RuntimeError: if called in Eager mode.

<hr>

*stateful*

None

<hr>

*submodules*

Sequence of all sub-modules.

    Submodules are modules which are properties of this module, or found as
    properties of modules which are properties of this module (and so on).

    ```
    >>> a = tf.Module()
    >>> b = tf.Module()
    >>> c = tf.Module()
    >>> a.b = b
    >>> b.c = c
    >>> list(a.submodules) == [b, c]
    True
    >>> list(b.submodules) == [c]
    True
    >>> list(c.submodules) == []
    True

    Returns:
    A sequence of all submodules.
```

<hr>

*supports_masking*

Whether this layer supports computing a mask using `compute_mask`.

<hr>

*trainable*

None

<hr>

*trainable_variables*

Sequence of trainable variables owned by this module and its submodules.

    Note: this method uses reflection to find variables on the current instance
    and submodules. For performance reasons you may wish to cache the result
    of calling this method if you don't expect the return value to change.

    Returns:
    A sequence of variables for the current module (sorted by attribute
    name) followed by variables from all submodules recursively (breadth
    first).

<hr>

*trainable_weights*

List of all trainable weights tracked by this layer.

    Trainable weights are updated via gradient descent during training.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
    A list of trainable variables.

<hr>

*updates*

None

<hr>

*variable_dtype*

Alias of `Layer.dtype`, the dtype of the weights.

<hr>

*variables*

Returns the list of all layer variables/weights.

    Alias of `self.weights`.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
    A list of variables.

<hr>

*weights*

Returns the list of all layer variables/weights.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
    A list of variables.

## OrdinalEarthMoversDistance

*OrdinalEarthMoversDistance(*args, **kwargs)*

Computes earth movers distance for ordinal labels.

### Methods

<hr>

*add_loss(losses, **kwargs)*

Add loss tensor(s), potentially dependent on layer inputs.

    Some losses (for instance, activity regularization losses) may be dependent
    on the inputs passed when calling a layer. Hence, when reusing the same
    layer on different inputs `a` and `b`, some entries in `layer.losses` may
    be dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    This method can be used inside a subclassed layer or model's `call`
    function, in which case `losses` should be a Tensor or list of Tensors.

    Example:

    ```python
    class MyLayer(tf.keras.layers.Layer):
    def call(self, inputs):
    self.add_loss(tf.abs(tf.reduce_mean(inputs)))
    return inputs
    ```

    This method can also be called directly on a Functional Model during
    construction. In this case, any loss Tensors passed to this Model must
    be symbolic and be able to be traced back to the model's `Input`s. These
    losses become part of the model's topology and are tracked in `get_config`.

    Example:

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    # Activity regularization.
    model.add_loss(tf.abs(tf.reduce_mean(x)))
    ```

    If this is not the case for your loss (if, for example, your loss references
    a `Variable` of one of the model's layers), you can wrap your loss in a
    zero-argument lambda. These losses are not tracked as part of the model's
    topology since they can't be serialized.

    Example:

    ```python
    inputs = tf.keras.Input(shape=(10,))
    d = tf.keras.layers.Dense(10)
    x = d(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    # Weight regularization.
    model.add_loss(lambda: tf.reduce_mean(d.kernel))
    ```

    Arguments:
    losses: Loss tensor, or list/tuple of tensors. Rather than tensors, losses
    may also be zero-argument callables which create a loss tensor.
**kwargs: Additional keyword arguments for backward compatibility.
    Accepted values:

inputs - Deprecated, will be automatically inferred.

<hr>

*add_metric(value, name=None, **kwargs)*

Adds metric tensor to the layer.

    This method can be used inside the `call()` method of a subclassed layer
    or model.

    ```python
    class MyMetricLayer(tf.keras.layers.Layer):
    def __init__(self):
    super(MyMetricLayer, self).__init__(name='my_metric_layer')
    self.mean = tf.keras.metrics.Mean(name='metric_1')

    def call(self, inputs):
    self.add_metric(self.mean(x))
    self.add_metric(tf.reduce_sum(x), name='metric_2')
    return inputs
    ```

    This method can also be called directly on a Functional Model during
    construction. In this case, any tensor passed to this Model must
    be symbolic and be able to be traced back to the model's `Input`s. These
    metrics become part of the model's topology and are tracked when you
    save the model via `save()`.

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.add_metric(math_ops.reduce_sum(x), name='metric_1')
    ```

    Note: Calling `add_metric()` with the result of a metric object on a
    Functional Model, as shown in the example below, is not supported. This is
    because we cannot trace the metric result tensor back to the model's inputs.

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.add_metric(tf.keras.metrics.Mean()(x), name='metric_1')
    ```

    Args:
    value: Metric tensor.
    name: String metric name.
**kwargs: Additional keyword arguments for backward compatibility.
    Accepted values:

`aggregation` - When the `value` tensor provided is not the result of
    calling a `keras.Metric` instance, it will be aggregated by default
    using a `keras.Metric.Mean`.

<hr>

*add_update(updates, inputs=None)*

Add update op(s), potentially dependent on layer inputs.

    Weight updates (for instance, the updates of the moving mean and variance
    in a BatchNormalization layer) may be dependent on the inputs passed
    when calling a layer. Hence, when reusing the same layer on
    different inputs `a` and `b`, some entries in `layer.updates` may be
    dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    This call is ignored when eager execution is enabled (in that case, variable
    updates are run on the fly and thus do not need to be tracked for later
    execution).

    Arguments:
    updates: Update op, or list/tuple of update ops, or zero-arg callable
    that returns an update op. A zero-arg callable should be passed in
    order to disable running the updates by setting `trainable=False`
    on this Layer, when executing in Eager mode.
    inputs: Deprecated, will be automatically inferred.

<hr>

*add_variable(*args, **kwargs)*

Deprecated, do NOT use! Alias for `add_weight`.

<hr>

*add_weight(name, shape=(), aggregation=<VariableAggregation.SUM: 1>, synchronization=<VariableSynchronization.ON_READ: 3>, initializer=None, dtype=None)*

Adds state variable. Only for use by subclasses.

<hr>

*apply(inputs, *args, **kwargs)*

Deprecated, do NOT use!

    This is an alias of `self.__call__`.

    Arguments:
    inputs: Input tensor(s).
    *args: additional positional arguments to be passed to `self.call`.
**kwargs: additional keyword arguments to be passed to `self.call`.


Returns:
    Output tensor(s).

<hr>

*build(input_shape)*

Creates the variables of the layer (optional, for subclass implementers).

    This is a method that implementers of subclasses of `Layer` or `Model`
    can override if they need a state-creation step in-between
    layer instantiation and layer call.

    This is typically used to create the weights of `Layer` subclasses.

    Arguments:
    input_shape: Instance of `TensorShape`, or list of instances of
    `TensorShape` if the layer expects a list of inputs
    (one instance per input).

<hr>

*call(inputs, **kwargs)*

This is where the layer's logic lives.

    Note here that `call()` method in `tf.keras` is little bit different
    from `keras` API. In `keras` API, you can pass support masking for
    layers as additional arguments. Whereas `tf.keras` has `compute_mask()`
    method to support masking.

    Arguments:
    inputs: Input tensor, or list/tuple of input tensors.
**kwargs: Additional keyword arguments. Currently unused.


Returns:
    A tensor or list/tuple of tensors.

<hr>

*compute_mask(inputs, mask=None)*

Computes an output mask tensor.

    Arguments:
    inputs: Tensor or list of tensors.
    mask: Tensor or list of tensors.

    Returns:
    None or a tensor (or list of tensors,
    one per output tensor of the layer).

<hr>

*compute_output_shape(input_shape)*

Computes the output shape of the layer.

    If the layer has not been built, this method will call `build` on the
    layer. This assumes that the layer will later be used with inputs that
    match the input shape provided here.

    Arguments:
    input_shape: Shape tuple (tuple of integers)
    or list of shape tuples (one per output tensor of the layer).
    Shape tuples can include None for free dimensions,
    instead of an integer.

    Returns:
    An input shape tuple.

<hr>

*compute_output_signature(input_signature)*

Compute the output tensor signature of the layer based on the inputs.

    Unlike a TensorShape object, a TensorSpec object contains both shape
    and dtype information for a tensor. This method allows layers to provide
    output dtype information if it is different from the input dtype.
    For any layer that doesn't implement this function,
    the framework will fall back to use `compute_output_shape`, and will
    assume that the output dtype matches the input dtype.

    Args:
    input_signature: Single TensorSpec or nested structure of TensorSpec
    objects, describing a candidate input for the layer.

    Returns:
    Single TensorSpec or nested structure of TensorSpec objects, describing
    how the layer would transform the provided input.

    Raises:
    TypeError: If input_signature contains a non-TensorSpec object.

<hr>

*count_params()*

Count the total number of scalars composing the weights.

    Returns:
    An integer count.

    Raises:
    ValueError: if the layer isn't yet built
    (in which case its weights aren't yet defined).

<hr>

*from_config(config)*

Creates a layer from its config.

    This method is the reverse of `get_config`,
    capable of instantiating the same layer from the config
    dictionary. It does not handle layer connectivity
    (handled by Network), nor weights (handled by `set_weights`).

    Arguments:
    config: A Python dictionary, typically the
    output of get_config.

    Returns:
    A layer instance.

<hr>

*get_config()*

Returns the serializable config of the metric.

<hr>

*get_input_at(node_index)*

Retrieves the input tensor(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A tensor (or list of tensors if the layer has multiple inputs).

    Raises:
    RuntimeError: If called in Eager mode.

<hr>

*get_input_mask_at(node_index)*

Retrieves the input mask tensor(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A mask tensor
    (or list of tensors if the layer has multiple inputs).

<hr>

*get_input_shape_at(node_index)*

Retrieves the input shape(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A shape tuple
    (or list of shape tuples if the layer has multiple inputs).

    Raises:
    RuntimeError: If called in Eager mode.

<hr>

*get_losses_for(inputs)*

Deprecated, do NOT use!

    Retrieves losses relevant to a specific set of inputs.

    Arguments:
    inputs: Input tensor or list/tuple of input tensors.

    Returns:
    List of loss tensors of the layer that depend on `inputs`.

<hr>

*get_output_at(node_index)*

Retrieves the output tensor(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A tensor (or list of tensors if the layer has multiple outputs).

    Raises:
    RuntimeError: If called in Eager mode.

<hr>

*get_output_mask_at(node_index)*

Retrieves the output mask tensor(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A mask tensor
    (or list of tensors if the layer has multiple outputs).

<hr>

*get_output_shape_at(node_index)*

Retrieves the output shape(s) of a layer at a given node.

    Arguments:
    node_index: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.

    Returns:
    A shape tuple
    (or list of shape tuples if the layer has multiple outputs).

    Raises:
    RuntimeError: If called in Eager mode.

<hr>

*get_updates_for(inputs)*

Deprecated, do NOT use!

    Retrieves updates relevant to a specific set of inputs.

    Arguments:
    inputs: Input tensor or list/tuple of input tensors.

    Returns:
    List of update ops of the layer that depend on `inputs`.

<hr>

*get_weights()*

Returns the current weights of the layer.

    The weights of a layer represent the state of the layer. This function
    returns both trainable and non-trainable weight values associated with this
    layer as a list of Numpy arrays, which can in turn be used to load state
    into similarly parameterized layers.

    For example, a Dense layer returns a list of two values-- per-output
    weights and the bias value. These can be used to set the weights of another
    Dense layer:

    ```
    >>> a = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(1.))
    >>> a_out = a(tf.convert_to_tensor([[1., 2., 3.]]))
    >>> a.get_weights()
    [array([[1.],
    [1.],
    [1.]], dtype=float32), array([0.], dtype=float32)]
    >>> b = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(2.))
    >>> b_out = b(tf.convert_to_tensor([[10., 20., 30.]]))
    >>> b.get_weights()
    [array([[2.],
    [2.],
    [2.]], dtype=float32), array([0.], dtype=float32)]
    >>> b.set_weights(a.get_weights())
    >>> b.get_weights()
    [array([[1.],
    [1.],
    [1.]], dtype=float32), array([0.], dtype=float32)]

    Returns:
    Weights values as a list of numpy arrays.
```

<hr>

*reset_state()*

Resets all of the metric state variables at the start of each epoch.

<hr>

*reset_states()*

Resets all of the metric state variables.

    This function is called between epochs/steps,
    when a metric is evaluated during training.

<hr>

*result()*

Computes and returns the metric value tensor.

    Result computation is an idempotent operation that simply calculates the
    metric value using the state variables.

<hr>

*set_weights(weights)*

Sets the weights of the layer, from Numpy arrays.

    The weights of a layer represent the state of the layer. This function
    sets the weight values from numpy arrays. The weight values should be
    passed in the order they are created by the layer. Note that the layer's
    weights must be instantiated before calling this function by calling
    the layer.

    For example, a Dense layer returns a list of two values-- per-output
    weights and the bias value. These can be used to set the weights of another
    Dense layer:

    ```
    >>> a = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(1.))
    >>> a_out = a(tf.convert_to_tensor([[1., 2., 3.]]))
    >>> a.get_weights()
    [array([[1.],
    [1.],
    [1.]], dtype=float32), array([0.], dtype=float32)]
    >>> b = tf.keras.layers.Dense(1,
    ...   kernel_initializer=tf.constant_initializer(2.))
    >>> b_out = b(tf.convert_to_tensor([[10., 20., 30.]]))
    >>> b.get_weights()
    [array([[2.],
    [2.],
    [2.]], dtype=float32), array([0.], dtype=float32)]
    >>> b.set_weights(a.get_weights())
    >>> b.get_weights()
    [array([[1.],
    [1.],
    [1.]], dtype=float32), array([0.], dtype=float32)]

    Arguments:
    weights: a list of Numpy arrays. The number
    of arrays and their shape must match
    number of the dimensions of the weights
    of the layer (i.e. it should match the
    output of `get_weights`).

    Raises:
    ValueError: If the provided weights list does not match the
    layer's specifications.
```

<hr>

*update_state(y_true, y_pred, sample_weight=None)*

Computes mean absolute error for ordinal labels.

    Args:
    y_true: Cumulatiuve logits from CondorOrdinal layer.
    y_pred: CondorOrdinal Encoded Labels.
    sample_weight (optional): Not implemented.

<hr>

*with_name_scope(method)*

Decorator to automatically enter the module name scope.

    ```
    >>> class MyModule(tf.Module):
    ...   @tf.Module.with_name_scope
    ...   def __call__(self, x):
    ...     if not hasattr(self, 'w'):
    ...       self.w = tf.Variable(tf.random.normal([x.shape[1], 3]))
    ...     return tf.matmul(x, self.w)

    Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
    names included the module name:

    >>> mod = MyModule()
    >>> mod(tf.ones([1, 2]))
    <tf.Tensor: shape=(1, 3), dtype=float32, numpy=..., dtype=float32)>
    >>> mod.w
    <tf.Variable 'my_module/Variable:0' shape=(2, 3) dtype=float32,
    numpy=..., dtype=float32)>

    Args:
    method: The method to wrap.

    Returns:
    The original method wrapped such that it enters the module's name scope.
```

### Properties

<hr>

*activity_regularizer*

Optional regularizer function for the output of this layer.

<hr>

*compute_dtype*

The dtype of the layer's computations.

    This is equivalent to `Layer.dtype_policy.compute_dtype`. Unless
    mixed precision is used, this is the same as `Layer.dtype`, the dtype of
    the weights.

    Layers automatically cast their inputs to the compute dtype, which causes
    computations and the output to be in the compute dtype as well. This is done
    by the base Layer class in `Layer.__call__`, so you do not have to insert
    these casts if implementing your own layer.

    Layers often perform certain internal computations in higher precision when
    `compute_dtype` is float16 or bfloat16 for numeric stability. The output
    will still typically be float16 or bfloat16 in such cases.

    Returns:
    The layer's compute dtype.

<hr>

*dtype*

The dtype of the layer weights.

    This is equivalent to `Layer.dtype_policy.variable_dtype`. Unless
    mixed precision is used, this is the same as `Layer.compute_dtype`, the
    dtype of the layer's computations.

<hr>

*dtype_policy*

The dtype policy associated with this layer.

    This is an instance of a `tf.keras.mixed_precision.Policy`.

<hr>

*dynamic*

Whether the layer is dynamic (eager-only); set in the constructor.

<hr>

*inbound_nodes*

Deprecated, do NOT use! Only for compatibility with external Keras.

<hr>

*input*

Retrieves the input tensor(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer.

    Returns:
    Input tensor or list of input tensors.

    Raises:
    RuntimeError: If called in Eager mode.
    AttributeError: If no inbound nodes are found.

<hr>

*input_mask*

Retrieves the input mask tensor(s) of a layer.

    Only applicable if the layer has exactly one inbound node,
    i.e. if it is connected to one incoming layer.

    Returns:
    Input mask tensor (potentially None) or list of input
    mask tensors.

    Raises:
    AttributeError: if the layer is connected to
    more than one incoming layers.

<hr>

*input_shape*

Retrieves the input shape(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer, or if all inputs
    have the same shape.

    Returns:
    Input shape, as an integer shape tuple
    (or list of shape tuples, one tuple per input tensor).

    Raises:
    AttributeError: if the layer has no defined input_shape.
    RuntimeError: if called in Eager mode.

<hr>

*input_spec*

`InputSpec` instance(s) describing the input format for this layer.

    When you create a layer subclass, you can set `self.input_spec` to enable
    the layer to run input compatibility checks when it is called.
    Consider a `Conv2D` layer: it can only be called on a single input tensor
    of rank 4. As such, you can set, in `__init__()`:

    ```python
    self.input_spec = tf.keras.layers.InputSpec(ndim=4)
    ```

    Now, if you try to call the layer on an input that isn't rank 4
    (for instance, an input of shape `(2,)`, it will raise a nicely-formatted
    error:

    ```
    ValueError: Input 0 of layer conv2d is incompatible with the layer:
    expected ndim=4, found ndim=1. Full shape received: [2]
    ```

    Input checks that can be specified via `input_spec` include:
    - Structure (e.g. a single input, a list of 2 inputs, etc)
    - Shape
    - Rank (ndim)
    - Dtype

    For more information, see `tf.keras.layers.InputSpec`.

    Returns:
    A `tf.keras.layers.InputSpec` instance, or nested structure thereof.

<hr>

*losses*

List of losses added using the `add_loss()` API.

    Variable regularization tensors are created when this property is accessed,
    so it is eager safe: accessing `losses` under a `tf.GradientTape` will
    propagate gradients back to the corresponding variables.

    Examples:

    ```
    >>> class MyLayer(tf.keras.layers.Layer):
    ...   def call(self, inputs):
    ...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
    ...     return inputs
    >>> l = MyLayer()
    >>> l(np.ones((10, 1)))
    >>> l.losses
    [1.0]

    >>> inputs = tf.keras.Input(shape=(10,))
    >>> x = tf.keras.layers.Dense(10)(inputs)
    >>> outputs = tf.keras.layers.Dense(1)(x)
    >>> model = tf.keras.Model(inputs, outputs)
    >>> # Activity regularization.
    >>> len(model.losses)
    0
    >>> model.add_loss(tf.abs(tf.reduce_mean(x)))
    >>> len(model.losses)
    1

    >>> inputs = tf.keras.Input(shape=(10,))
    >>> d = tf.keras.layers.Dense(10, kernel_initializer='ones')
    >>> x = d(inputs)
    >>> outputs = tf.keras.layers.Dense(1)(x)
    >>> model = tf.keras.Model(inputs, outputs)
    >>> # Weight regularization.
    >>> model.add_loss(lambda: tf.reduce_mean(d.kernel))
    >>> model.losses
    [<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]

    Returns:
    A list of tensors.
```

<hr>

*metrics*

List of metrics added using the `add_metric()` API.

    Example:

    ```
    >>> input = tf.keras.layers.Input(shape=(3,))
    >>> d = tf.keras.layers.Dense(2)
    >>> output = d(input)
    >>> d.add_metric(tf.reduce_max(output), name='max')
    >>> d.add_metric(tf.reduce_min(output), name='min')
    >>> [m.name for m in d.metrics]
    ['max', 'min']

    Returns:
    A list of `Metric` objects.
```

<hr>

*name*

Name of the layer (string), set in the constructor.

<hr>

*name_scope*

Returns a `tf.name_scope` instance for this class.

<hr>

*non_trainable_variables*

None

<hr>

*non_trainable_weights*

List of all non-trainable weights tracked by this layer.

    Non-trainable weights are *not* updated during training. They are expected
    to be updated manually in `call()`.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
    A list of non-trainable variables.

<hr>

*outbound_nodes*

Deprecated, do NOT use! Only for compatibility with external Keras.

<hr>

*output*

Retrieves the output tensor(s) of a layer.

    Only applicable if the layer has exactly one output,
    i.e. if it is connected to one incoming layer.

    Returns:
    Output tensor or list of output tensors.

    Raises:
    AttributeError: if the layer is connected to more than one incoming
    layers.
    RuntimeError: if called in Eager mode.

<hr>

*output_mask*

Retrieves the output mask tensor(s) of a layer.

    Only applicable if the layer has exactly one inbound node,
    i.e. if it is connected to one incoming layer.

    Returns:
    Output mask tensor (potentially None) or list of output
    mask tensors.

    Raises:
    AttributeError: if the layer is connected to
    more than one incoming layers.

<hr>

*output_shape*

Retrieves the output shape(s) of a layer.

    Only applicable if the layer has one output,
    or if all outputs have the same shape.

    Returns:
    Output shape, as an integer shape tuple
    (or list of shape tuples, one tuple per output tensor).

    Raises:
    AttributeError: if the layer has no defined output shape.
    RuntimeError: if called in Eager mode.

<hr>

*stateful*

None

<hr>

*submodules*

Sequence of all sub-modules.

    Submodules are modules which are properties of this module, or found as
    properties of modules which are properties of this module (and so on).

    ```
    >>> a = tf.Module()
    >>> b = tf.Module()
    >>> c = tf.Module()
    >>> a.b = b
    >>> b.c = c
    >>> list(a.submodules) == [b, c]
    True
    >>> list(b.submodules) == [c]
    True
    >>> list(c.submodules) == []
    True

    Returns:
    A sequence of all submodules.
```

<hr>

*supports_masking*

Whether this layer supports computing a mask using `compute_mask`.

<hr>

*trainable*

None

<hr>

*trainable_variables*

Sequence of trainable variables owned by this module and its submodules.

    Note: this method uses reflection to find variables on the current instance
    and submodules. For performance reasons you may wish to cache the result
    of calling this method if you don't expect the return value to change.

    Returns:
    A sequence of variables for the current module (sorted by attribute
    name) followed by variables from all submodules recursively (breadth
    first).

<hr>

*trainable_weights*

List of all trainable weights tracked by this layer.

    Trainable weights are updated via gradient descent during training.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
    A list of trainable variables.

<hr>

*updates*

None

<hr>

*variable_dtype*

Alias of `Layer.dtype`, the dtype of the weights.

<hr>

*variables*

Returns the list of all layer variables/weights.

    Alias of `self.weights`.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
    A list of variables.

<hr>

*weights*

Returns the list of all layer variables/weights.

    Note: This will not track the weights of nested `tf.Modules` that are not
    themselves Keras layers.

    Returns:
    A list of variables.

