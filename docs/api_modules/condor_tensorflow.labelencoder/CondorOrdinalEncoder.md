## CondorOrdinalEncoder

*CondorOrdinalEncoder(nclasses=0, categories='auto', dtype=<class 'numpy.int32'>, handle_unknown='error', unknown_value=None)*

Base class for all estimators in scikit-learn.

**Notes**

All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
arguments (no ``*args`` or ``**kwargs``).

### Methods

<hr>

*fit(X, y=None)*

Fit the CondorOrdinalEncoder to X.

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

Transform X to ordinal arrays.

**Parameters**

- `X` : array-like of shape (n_samples, n_features)

    The data to encode.

**Returns**

- `X_out` : ndarray of shape (n_samples, n_features)

    Transformed input.

### Properties

