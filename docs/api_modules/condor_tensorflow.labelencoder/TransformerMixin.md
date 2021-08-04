## TransformerMixin

*TransformerMixin()*

Mixin class for all transformers in scikit-learn.

### Methods

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

### Properties

