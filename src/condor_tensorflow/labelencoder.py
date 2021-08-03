import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

class CondorOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, nclasses=0,categories='auto', dtype=np.int32,
                 handle_unknown='error', unknown_value=None):
        self.nclasses=nclasses
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value

    def fit(self, X, y=None):
        """
        Fit the CondorOrdinalEncoder to X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.
        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
        Returns
        -------
        self
        """
        if self.nclasses > 0:
            pass # expecting 0,1,...,nclasses-1
        else:
            self._enc = OrdinalEncoder(categories=self.categories,
                                      dtype=self.dtype,
                                      handle_unknown=self.handle_unknown,
                                      unknown_value=self.unknown_value)
            if len(X.shape)==1:
                X=X.reshape(-1,1)
            self._enc.fit(X)
        return self

    def transform(self, X):
        """
        Transform X to ordinal arrays.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.
        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed input.
        """
        if self.nclasses == 0:
            if len(X.shape)==1:
                X=X.reshape(-1,1)
                X = np.array(self._enc.transform(X)[:,0],
                             dtype=self.dtype)
            self.nclasses = len(self._enc.categories_[0])
        else:
            X = np.array(X,dtype=self.dtype)

        # now X always has values 0,1,...,nclasses-1
        # first make one-hot encoding
        X_out = np.zeros((X.shape[0],self.nclasses))
        X_out[np.arange(X.size),X] = 1

        #now drop first column
        X_out = X_out[:,1:]

        # and use cumsum to fill
        X_out = np.flip(np.flip(X_out,1).cumsum(axis=1),1)
        return X_out
