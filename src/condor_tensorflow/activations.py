import tensorflow as tf

@tf.function
def ordinal_softmax(x):
  """ Convert the ordinal logit output of CondorOrdinal() to label probabilities.

    Parameters
    ----------
    x: tf.Tensor, shape=(num_samples,num_classes-1)
        Logit output of the final Dense(num_classes-1) layer.

    Returns
    ----------
    probs_tensor: tf.Tensor, shape=(num_samples, num_classes)
        Probabilities of each class (columns) for each
        sample (rows).

    Examples
    ----------
    >>> condor.ordinal_softmax(tf.constant([[-1.,1],[-2,2]]))
    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[0.7310586 , 0.07232949, 0.19661194],
           [0.8807971 , 0.01420934, 0.10499357]], dtype=float32)>
  """

  # Number of columns is the number of classes - 1
  num_classes = x.shape[1] + 1

  # Convert the ordinal logits into cumulative probabilities.
  cum_probs = tf.math.cumprod(tf.math.sigmoid(x), axis = 1)#tf.map_fn(tf.math.sigmoid, x)

  # Create a list of tensors.
  probs = []

  # First, get probability predictions out of the cumulative logits.
  # Column 0 is Probability that y > 0, so Pr(y = 0) = 1 - Pr(y > 0)
  # Pr(Y = 0) = 1 - s(logit for column 0)
  probs.append(1. - cum_probs[:, 0])


  # For the other columns, the probability is:
  # Pr(y = k) = Pr(y > k) - Pr(y > k - 1)
  if num_classes > 2:
    for val in range(1, num_classes - 1):
      probs.append(cum_probs[:, val - 1] - cum_probs[:, val])

  # Special handling of the maximum label value.
  probs.append(cum_probs[:, num_classes - 2])

  # Combine as columns into a new tensor.
  probs_tensor = tf.concat(tf.transpose(probs), axis = 1)

  return probs_tensor
