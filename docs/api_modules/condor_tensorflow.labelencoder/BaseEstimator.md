## BaseEstimator

*BaseEstimator()*

Base class for all estimators in scikit-learn.

**Notes**

All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
arguments (no ``*args`` or ``**kwargs``).

### Methods

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

### Properties

