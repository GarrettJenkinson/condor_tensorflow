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

