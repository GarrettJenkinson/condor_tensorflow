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
      >>> ordinal_softmax(tf.constant([[-1.,1],[-2,2]]))
      <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
      array([[0.7310586 , 0.07232949, 0.19661194],
             [0.8807971 , 0.01420934, 0.10499357]], dtype=float32)>
    """

    # Convert the ordinal logits into cumulative probabilities.
    cum_probs = tf.concat(
            [tf.ones((tf.shape(x)[0], 1), dtype=tf.float32),
             tf.math.cumprod(tf.math.sigmoid(tf.cast(
                                  x, tf.float32)), axis=1), 
             tf.zeros((tf.shape(x)[0], 1), dtype=tf.float32)
            ], 1)

    return cum_probs[:, 0:-1] - cum_probs[:, 1:]
