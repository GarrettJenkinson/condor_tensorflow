import tensorflow as tf
from tensorflow.keras import backend as K


class OrdinalMeanAbsoluteError(tf.keras.metrics.Metric):
    """Computes mean absolute error for ordinal labels."""

    def __init__(self, name="mean_absolute_error_labels",
                 **kwargs):
        """Creates a `OrdinalMeanAbsoluteError` instance."""
        super().__init__(name=name, **kwargs)
        self.maes = self.add_weight(name='maes', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Computes mean absolute error for ordinal labels.

        Args:
          y_true: Cumulatiuve logits from CondorOrdinal layer.
          y_pred: CondorOrdinal Encoded Labels.
          sample_weight (optional): Not implemented.
        """

        if sample_weight:
            raise NotImplementedError

        # Predict the label as in Cao et al. - using cumulative probabilities
        cum_probs = tf.math.cumprod(
            tf.math.sigmoid(y_pred),
            axis=1)  # tf.map_fn(tf.math.sigmoid, y_pred)

        # Calculate the labels using the style of Cao et al.
        above_thresh = tf.map_fn(
            lambda x: tf.cast(
                x > 0.5,
                tf.float32),
            cum_probs)

        # Sum across columns to estimate how many cumulative thresholds are
        # passed.
        labels_v2 = tf.reduce_sum(above_thresh, axis=1)

        y_true = tf.cast(tf.reduce_sum(y_true, axis=1), y_pred.dtype)

        # remove all dimensions of size 1 (e.g., from [[1], [2]], to [1, 2])
        y_true = tf.squeeze(y_true)

        self.maes.assign_add(tf.reduce_sum(tf.abs(y_true - labels_v2)))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.maes, self.count)

    def reset_state(self):
        """Resets all of the metric state variables at the start of each epoch."""
        self.maes.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        """Returns the serializable config of the metric."""
        config = {}
        base_config = super().get_config()
        return {**base_config, **config}


class SparseOrdinalMeanAbsoluteError(OrdinalMeanAbsoluteError):
    """Computes mean absolute error for ordinal labels."""

    def __init__(self, name="mean_absolute_error_labels",
                 **kwargs):
        """Creates a `OrdinalMeanAbsoluteError` instance."""
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Computes mean absolute error for ordinal labels.

        Args:
          y_true: Cumulatiuve logits from CondorOrdinal layer.
          y_pred: CondorOrdinal Encoded Labels.
          sample_weight (optional): Not implemented.
        """

        if sample_weight:
            raise NotImplementedError

        # Predict the label as in Cao et al. - using cumulative probabilities
        cum_probs = tf.math.cumprod(
            tf.math.sigmoid(y_pred),
            axis=1)  # tf.map_fn(tf.math.sigmoid, y_pred)

        # Calculate the labels using the style of Cao et al.
        above_thresh = tf.map_fn(
            lambda x: tf.cast(
                x > 0.5,
                tf.float32),
            cum_probs)

        # Sum across columns to estimate how many cumulative thresholds are
        # passed.
        labels_v2 = tf.reduce_sum(above_thresh, axis=1)

        y_true = tf.cast(y_true, y_pred.dtype)

        # remove all dimensions of size 1 (e.g., from [[1], [2]], to [1, 2])
        y_true = tf.squeeze(y_true)

        self.maes.assign_add(tf.reduce_sum(tf.abs(y_true - labels_v2)))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

class OrdinalAccuracy(tf.keras.metrics.Metric):
    """Computes accuracy for ordinal labels (tolerance is allowed rank
    distance to be considered 'correct' predictions)."""

    def __init__(self, name=None,
                 tolerance=0,
                 **kwargs):
        """Creates a `OrdinalAccuracy` instance."""
        if name is not None:
            super().__init__(name=name, **kwargs)
        else:
            super().__init__(name="ordinal_accuracy_tol"+str(tolerance),
                             **kwargs)
        self.accs = self.add_weight(name='accs', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.tolerance = tolerance

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Computes accuracy for ordinal labels.

        Args:
          y_true: Cumulatiuve logits from CondorOrdinal layer.
          y_pred: CondorOrdinal Encoded Labels.
          sample_weight (optional): Not implemented.
        """

        if sample_weight:
            raise NotImplementedError

        # Predict the label as in Cao et al. - using cumulative probabilities
        cum_probs = tf.math.cumprod(
            tf.math.sigmoid(y_pred),
            axis=1)  # tf.map_fn(tf.math.sigmoid, y_pred)

        # Calculate the labels using the style of Cao et al.
        above_thresh = tf.map_fn(
            lambda x: tf.cast(
                x > 0.5,
                tf.float32),
            cum_probs)

        # Sum across columns to estimate how many cumulative thresholds are
        # passed.
        labels_v2 = tf.reduce_sum(above_thresh, axis=1)

        y_true = tf.cast(tf.reduce_sum(y_true, axis=1), y_pred.dtype)

        # remove all dimensions of size 1 (e.g., from [[1], [2]], to [1, 2])
        y_true = tf.squeeze(y_true)

        self.accs.assign_add(tf.reduce_sum(tf.cast(tf.less_equal(
            tf.abs(y_true-labels_v2),tf.cast(self.tolerance,y_pred.dtype)),
            y_pred.dtype)))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.accs, self.count)

    def reset_state(self):
        """Resets all of the metric state variables at the start of each epoch."""
        self.accs.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        """Returns the serializable config of the metric."""
        config = {'tolerance': self.tolerance}
        base_config = super().get_config()
        return {**base_config, **config}


class SparseOrdinalAccuracy(OrdinalAccuracy):
    """Computes accuracy for ordinal labels (tolerance is allowed rank
    distance to be considered 'correct' predictions)."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Computes accuracy for ordinal labels.

        Args:
          y_true: Cumulatiuve logits from CondorOrdinal layer.
          y_pred: CondorOrdinal Encoded Labels.
          sample_weight (optional): Not implemented.
        """

        if sample_weight:
            raise NotImplementedError

        # Predict the label as in Cao et al. - using cumulative probabilities
        cum_probs = tf.math.cumprod(
            tf.math.sigmoid(y_pred),
            axis=1)  # tf.map_fn(tf.math.sigmoid, y_pred)

        # Calculate the labels using the style of Cao et al.
        above_thresh = tf.map_fn(
            lambda x: tf.cast(
                x > 0.5,
                tf.float32),
            cum_probs)

        # Sum across columns to estimate how many cumulative thresholds are
        # passed.
        labels_v2 = tf.reduce_sum(above_thresh, axis=1)

        y_true = tf.cast(y_true, y_pred.dtype)

        # remove all dimensions of size 1 (e.g., from [[1], [2]], to [1, 2])
        y_true = tf.squeeze(y_true)

        self.accs.assign_add(tf.reduce_sum(tf.cast(tf.less_equal(
            tf.abs(y_true-labels_v2),tf.cast(self.tolerance,y_pred.dtype)),
            y_pred.dtype)))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

