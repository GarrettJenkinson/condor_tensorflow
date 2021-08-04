## OrdinalEncoder

*OrdinalEncoder(*, categories='auto', dtype=<class 'numpy.float64'>, handle_unknown='error', unknown_value=None)*

Encode categorical features as an integer array.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are converted to ordinal integers. This results in
    a single column of integers (0 to n_categories - 1) per feature.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

    .. versionadded:: 0.20

**Parameters**

- `categories` : 'auto' or a list of array-like, default='auto'

    Categories (unique values) per feature:


- `- 'auto'` : Determine categories automatically from the training data.


- `- list` : ``categories[i]`` holds the categories expected in the ith

    column. The passed categories should not mix strings and numeric
    values, and should be sorted in case of numeric values.

    The used categories can be found in the ``categories_`` attribute.


- `dtype` : number type, default np.float64

    Desired dtype of output.


- `handle_unknown` : {'error', 'use_encoded_value'}, default='error'

    When set to 'error' an error will be raised in case an unknown
    categorical feature is present during transform. When set to
    'use_encoded_value', the encoded value of unknown categories will be
    set to the value given for the parameter `unknown_value`. In
    :meth:`inverse_transform`, an unknown category will be denoted as None.

    .. versionadded:: 0.24


- `unknown_value` : int or np.nan, default=None

    When the parameter handle_unknown is set to 'use_encoded_value', this
    parameter is required and will set the encoded value of unknown
    categories. It has to be distinct from the values used to encode any of
    the categories in `fit`. If set to np.nan, the `dtype` parameter must
    be a float dtype.

    .. versionadded:: 0.24

**Attributes**

- `categories_` : list of arrays

    The categories of each feature determined during ``fit`` (in order of
    the features in X and corresponding with the output of ``transform``).
    This does not include categories that weren't seen during ``fit``.

**See Also**

- `OneHotEncoder` : Performs a one-hot encoding of categorical features.


- `LabelEncoder` : Encodes target labels with values between 0 and

    ``n_classes-1``.

**Examples**

Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to an ordinal encoding.

    ```
    >>> from sklearn.preprocessing import OrdinalEncoder
    >>> enc = OrdinalEncoder()
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    OrdinalEncoder()
    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 3], ['Male', 1]])
    array([[0., 2.],
    [1., 0.]])

    >>> enc.inverse_transform([[1, 0], [0, 1]])
    array([['Male', 1],
    ['Female', 2]], dtype=object)
```

### Methods

<hr>

*fit(X, y=None)*

Fit the OrdinalEncoder to X.

**Parameters**

- `X` : array-like of shape (n_samples, n_features)

    The data to determine the categories of each feature.


- `y` : None

    Ignored. This parameter exists only for compatibility with
    :class:`~sklearn.pipeline.Pipeline`.

**Returns**

self

<hr>

*fit_transform(X, y=None, **fit_params)*

Fit to data, then transform it.

    Fits transformer to `X` and `y` with optional parameters `fit_params`
    and returns a transformed version of `X`.

**Parameters**

- `X` : array-like of shape (n_samples, n_features)

    Input samples.


- `y` :  array-like of shape (n_samples,) or (n_samples, n_outputs),                 default=None

    Target values (None for unsupervised transformations).


- `**fit_params` : dict

    Additional fit parameters.

**Returns**

- `X_new` : ndarray array of shape (n_samples, n_features_new)

    Transformed array.

<hr>

*get_params(deep=True)*

Get parameters for this estimator.

**Parameters**

- `deep` : bool, default=True

    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

**Returns**

- `params` : dict

    Parameter names mapped to their values.

<hr>

*inverse_transform(X)*

Convert the data back to the original representation.

**Parameters**

- `X` : {array-like, sparse matrix} of shape (n_samples, n_features)

    The transformed data.

**Returns**

- `X_tr` : ndarray of shape (n_samples, n_features)

    Inverse transformed array.

<hr>

*set_params(**params)*

Set the parameters of this estimator.

    The method works on simple estimators as well as on nested objects
    (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
    parameters of the form ``<component>__<parameter>`` so that it's
    possible to update each component of a nested object.

**Parameters**

- `**params` : dict

    Estimator parameters.

**Returns**

- `self` : estimator instance

    Estimator instance.

<hr>

*transform(X)*

Transform X to ordinal codes.

**Parameters**

- `X` : array-like of shape (n_samples, n_features)

    The data to encode.

**Returns**

- `X_out` : ndarray of shape (n_samples, n_features)

    Transformed input.

### Properties

