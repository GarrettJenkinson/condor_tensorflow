from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf
from .activations import ordinal_softmax

# The outer function is a constructor to create a loss function using a
# certain number of classes.

class CondorNegLogLikelihood(tf.keras.losses.Loss):

    def __init__(self,
                 from_type="ordinal_logits",
                 name="ordinal_nll",
                 **kwargs):
        """ Negative log likelihood loss designed for ordinal outcomes.

        Parameters
        ----------
        from_type: one of "ordinal_logits" (default), or "probs".
          Ordinal logits are the output of a Dense(num_classes-1) layer with no activation.
          (Not yet implemented) Probs are the probability outputs of a softmax or ordinal_softmax layer.

        Returns
        ----------
        loss: tf.Tensor, shape=(num_samples,)
            Loss vector, note that tensorflow will reduce it to a single number
            automatically.
        """
        self.from_type = from_type

        super().__init__(name=name, **kwargs)

    # Modifed from: https://github.com/tensorflow/tensorflow/blob/6dcd6fcea73ad613e78039bd1f696c35e63abb32/tensorflow/python/ops/nn_impl.py#L112-L148
    def ordinal_loss(self, logits, labels, name=None):
        """ Negative log likelihood loss function designed for ordinal outcomes.

        Parameters
        ----------
        logits: tf.Tensor, shape=(num_samples,num_classes-1)
            Logit output of the final Dense(num_classes-1) layer.

        levels: tf.Tensor, shape=(num_samples, num_classes-1)
            Encoded lables provided by CondorOrdinalEncoder.

        Returns
        ----------
        loss: tf.Tensor, shape=(num_samples,)
            Loss vector, note that tensorflow will reduce it to a single number
            automatically.
        """
        with ops.name_scope(name, "logistic_loss", [logits, labels]) as name:
          if isinstance(logits,tf.Tensor):
              logits = tf.cast(logits,dtype=tf.float32,name="logits")
          else:
              logits = ops.convert_to_tensor(logits, dtype=tf.float32,name="logits")
          if isinstance(labels,tf.Tensor):
              labs = tf.cast(labels,dtype=tf.float32,name="labs")
              piLab = tf.concat([tf.ones((tf.shape(labs)[0],1)),labs[:,:-1]],
                                axis=1,name="piLab")
          else:
              labs = ops.convert_to_tensor(labels, dtype=tf.float32,name="labs")
              piLab = tf.concat([tf.ones((tf.shape(labs)[0],1)),labs[:,:-1]],
                                axis=1,name="piLab")

          # The logistic loss formula from above is
          #   x - x * z + log(1 + exp(-x))
          # For x < 0, a more numerically stable formula is
          #   -x * z + log(1 + exp(x))
          # Note that these two expressions can be combined into the following:
          #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
          # To allow computing gradients at zero, we define custom versions of max and
          # abs functions.
          zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
          cond = (logits >= zeros)
          cond2 = (piLab > zeros)
          relu_logits = array_ops.where(cond, logits, zeros)
          neg_abs_logits = array_ops.where(cond, -logits, logits)
          temp = math_ops.add(relu_logits - logits * labs,
                              math_ops.log1p(math_ops.exp(neg_abs_logits)))
          return tf.math.reduce_sum(array_ops.where(cond2, temp, zeros),
                                    axis=1,name=name)

    # Following https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss
    def call(self, y_true, y_pred):

        # Ensure that y_true is the same type as y_pred (presumably a float).
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        # get number of classes
        num_classes = tf.shape(y_pred)[1]+1

        # we are not sparse here, so labels are encoded already
        tf_levels = y_true

        if self.from_type == "ordinal_logits":
            return self.ordinal_loss(y_pred, tf_levels)
        elif self.from_type == "probs":
            raise Exception("not yet implemented")
        elif self.from_type == "logits":
            raise Exception("not yet implemented")
        else:
            raise Exception("Unknown from_type value " + self.from_type +
                            " in CondorNegLogLikelihood()")

    def get_config(self):
        base_config = super().get_config()
        config = {
            "from_type": self.from_type,
        }
        return {**base_config, **config}


# The outer function is a constructor to create a loss function using a
# certain number of classes.
class SparseCondorNegLogLikelihood(CondorNegLogLikelihood):

    def __init__(self,
                 from_type="ordinal_logits",
                 name="ordinal_negLogLikeloss",
                 **kwargs):
        """ Negative log likelihood loss designed for ordinal outcomes.

        Parameters
        ----------
        from_type: one of "ordinal_logits" (default), or "probs".
          Ordinal logits are the output of a Dense(num_classes-1) layer with no activation.
          (Not yet implemented) Probs are the probability outputs of a softmax or ordinal_softmax layer.

        Returns
        ----------
        loss: tf.Tensor, shape=(num_samples,)
            Loss vector, note that tensorflow will reduce it to a single number
            automatically.
        """
        super().__init__(name=name,
                         from_type=from_type,
                         **kwargs)

    def label_to_levels(self, label):
        # Original code that we are trying to replicate:
        # levels = [1] * label + [0] * (self.num_classes - 1 - label)
        label_vec = tf.repeat(1, tf.cast(tf.squeeze(label), tf.int32))

        # This line requires that label values begin at 0. If they start at a higher
        # value it will yield an error.
        num_zeros = self.num_classes - 1 - tf.cast(tf.squeeze(label), tf.int32)

        zero_vec = tf.zeros(shape=(num_zeros), dtype=tf.int32)

        levels = tf.concat([label_vec, zero_vec], axis=0)

        return tf.cast(levels, tf.float32)

    # Following https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss
    def call(self, y_true, y_pred):

        # Ensure that y_true is the same type as y_pred (presumably a float).
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        # get number of classes
        self.num_classes = tf.shape(y_pred)[1]+1

        # Convert each true label to a vector of ordinal level indicators.
        tf_levels = tf.map_fn(self.label_to_levels, y_true)

        if self.from_type == "ordinal_logits":
            return self.ordinal_loss(y_pred, tf_levels)
        elif self.from_type == "probs":
            raise Exception("not yet implemented")
        elif self.from_type == "logits":
            raise Exception("not yet implemented")
        else:
            raise Exception("Unknown from_type value " + self.from_type +
                            " in SparseCondorNegLogLikelihood()")

class CondorOrdinalCrossEntropy(tf.keras.losses.Loss):

    def __init__(self,
                 importance_weights=None,
                 from_type="ordinal_logits",
                 name="ordinal_crossent",
                 **kwargs):
        """ Cross-entropy loss designed for ordinal outcomes.

        Parameters
        ----------
        importance_weights: tf or np array of floats, shape(numclasses-1,)
            (Optional) importance weights for each binary classification task.

        from_type: one of "ordinal_logits" (default), or "probs".
          Ordinal logits are the output of a Dense(num_classes-1) layer with no activation.
          (Not yet implemented) Probs are the probability outputs of a softmax or ordinal_softmax layer.

        Returns
        ----------
        loss: tf.Tensor, shape=(num_samples,)
            Loss vector, note that tensorflow will reduce it to a single number
            automatically.
        """
        self.importance_weights = importance_weights
        self.from_type = from_type

        super().__init__(name=name, **kwargs)

    def ordinal_loss(self, logits, levels, importance):
        """ Cross-entropy loss function designed for ordinal outcomes.

        Parameters
        ----------
        logits: tf.Tensor, shape=(num_samples,num_classes-1)
            Logit output of the final Dense(num_classes-1) layer.

        levels: tf.Tensor, shape=(num_samples, num_classes-1)
            Encoded lables provided by CondorOrdinalEncoder.

        importance_weights: tf or np array of floats, shape(numclasses-1,)
            Importance weights for each binary classification task.

        Returns
        ----------
        loss: tf.Tensor, shape=(num_samples,)
            Loss vector, note that tensorflow will reduce it to a single number
            automatically.
        """
        logprobs = tf.math.cumsum(tf.math.log_sigmoid(logits), axis=1)
        eps = tf.keras.backend.epsilon()
        val = (-tf.reduce_sum(importance * (logprobs * levels + \
               (tf.math.log(1 - tf.math.exp(logprobs) + eps) * (1 - levels))), axis=1))
        return val

    # Following https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss
    def call(self, y_true, y_pred):

        # Ensure that y_true is the same type as y_pred (presumably a float).
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        # get number of classes
        num_classes = tf.shape(y_pred)[1]+1

        # we are not sparse here, so labels are encoded already
        tf_levels = y_true

        if self.importance_weights is None:
            importance_weights = tf.ones(num_classes-1,
                                         dtype=tf.float32)
        else:
            importance_weights = tf.cast(
                self.importance_weights, dtype=tf.float32)

        if self.from_type == "ordinal_logits":
            return self.ordinal_loss(y_pred, tf_levels, importance_weights)
        elif self.from_type == "probs":
            raise Exception("not yet implemented")
        elif self.from_type == "logits":
            raise Exception("not yet implemented")
        else:
            raise Exception("Unknown from_type value " + self.from_type +
                            " in CondorOrdinalCrossEntropy()")

    def get_config(self):
        base_config = super().get_config()
        config = {
            "importance_weights": self.importance_weights,
            "from_type": self.from_type,
        }
        return {**base_config, **config}


# The outer function is a constructor to create a loss function using a
# certain number of classes.
class SparseCondorOrdinalCrossEntropy(CondorOrdinalCrossEntropy):

    def __init__(self,
                 importance_weights=None,
                 from_type="ordinal_logits",
                 name="ordinal_crossent",
                 **kwargs):
        """ Cross-entropy loss designed for ordinal outcomes.

        Parameters
        ----------
        importance_weights: tf or np array of floats, shape(numclasses-1,)
            (Optional) importance weights for each binary classification task.

        from_type: one of "ordinal_logits" (default), or "probs".
          Ordinal logits are the output of a Dense(num_classes-1) layer with no activation.
          (Not yet implemented) Probs are the probability outputs of a softmax or ordinal_softmax layer.

        Returns
        ----------
        loss: tf.Tensor, shape=(num_samples,)
            Loss vector, note that tensorflow will reduce it to a single number
            automatically.
        """
        super().__init__(name=name,
                         importance_weights=importance_weights,
                         from_type=from_type,
                         **kwargs)

    def label_to_levels(self, label):
        # Original code that we are trying to replicate:
        # levels = [1] * label + [0] * (self.num_classes - 1 - label)
        label_vec = tf.repeat(1, tf.cast(tf.squeeze(label), tf.int32))

        # This line requires that label values begin at 0. If they start at a higher
        # value it will yield an error.
        num_zeros = self.num_classes - 1 - tf.cast(tf.squeeze(label), tf.int32)

        zero_vec = tf.zeros(shape=(num_zeros), dtype=tf.int32)

        levels = tf.concat([label_vec, zero_vec], axis=0)

        return tf.cast(levels, tf.float32)

    # Following https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss

    def call(self, y_true, y_pred):

        # Ensure that y_true is the same type as y_pred (presumably a float).
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        # get number of classes
        self.num_classes = tf.shape(y_pred)[1]+1

        # Convert each true label to a vector of ordinal level indicators.
        tf_levels = tf.map_fn(self.label_to_levels, y_true)

        if self.importance_weights is None:
            importance_weights = tf.ones(
                self.num_classes - 1, dtype=tf.float32)
        else:
            importance_weights = tf.cast(
                self.importance_weights, dtype=tf.float32)

        if self.from_type == "ordinal_logits":
            return self.ordinal_loss(y_pred, tf_levels, importance_weights)
        elif self.from_type == "probs":
            raise Exception("not yet implemented")
        elif self.from_type == "logits":
            raise Exception("not yet implemented")
        else:
            raise Exception("Unknown from_type value " + self.from_type +
                            " in CondorOrdinalCrossEntropy()")



class OrdinalEarthMoversDistance(tf.keras.losses.Loss):
    """Computes earth movers distance for ordinal labels."""

    def __init__(self, name="earth_movers_distance",
                 **kwargs):
        """Creates a `OrdinalEarthMoversDistance` instance."""
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        """Computes mean absolute error for ordinal labels.

        Args:
          y_true: Cumulatiuve logits from CondorOrdinal layer.
          y_pred: CondorOrdinal Encoded Labels.
        """

        # Ensure that y_true is the same type as y_pred (presumably a float).
        y_pred = tf.convert_to_tensor(y_pred)

        # basic setup
        cum_probs = ordinal_softmax(y_pred)
        num_classes = tf.shape(cum_probs)[1]

        y_true = tf.cast(tf.reduce_sum(y_true, axis=1), y_pred.dtype)

        # remove all dimensions of size 1 (e.g., from [[1], [2]], to [1, 2])
        #y_true = tf.squeeze(y_true)

        y_dist = tf.map_fn(
            fn=lambda y: tf.abs(
                y - tf.range(num_classes,dtype=y_pred.dtype)),
            elems=y_true)

        vals = tf.reduce_sum(tf.math.multiply(y_dist,cum_probs),axis=1)
        return vals

    def get_config(self):
        """Returns the serializable config of the metric."""
        base_config = super().get_config()
        return {**base_config}


class SparseOrdinalEarthMoversDistance(OrdinalEarthMoversDistance):
    """Computes earth movers distance for ordinal labels."""

    def __init__(self, **kwargs):
        """Creates a `SparseOrdinalEarthMoversDistance` instance."""
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """Computes mean absolute error for ordinal labels.

        Args:
          y_true: Cumulatiuve logits from CondorOrdinal layer.
          y_pred: Sparse Labels with values in {0,1,...,num_classes-1}
        """
        # basic set up
        cum_probs = ordinal_softmax(y_pred)
        num_classes = tf.shape(cum_probs)[1]
        y_true = tf.cast(y_true, y_pred.dtype)

        # remove all dimensions of size 1 (e.g., from [[1], [2]], to [1, 2])
        #y_true = tf.squeeze(y_true)

        # each row has distance to true label
        y_dist = tf.map_fn(
            fn=lambda y: tf.abs(y - tf.range(num_classes,
                                dtype=y_pred.dtype)),
            elems=y_true)

        # pointwise multiplication by the class probabilities, row-wise sums
        vals = tf.reduce_sum(tf.math.multiply(y_dist,cum_probs),axis=1)
        return vals
