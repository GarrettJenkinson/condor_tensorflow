
<img src="./img/condor.png" width=300>

**CONDOR tensorflow implementation for ordinal regression with deep neural networks.**

[![PyPi version](https://pypip.in/v/condor-tensorflow/badge.png)](https://pypi.org/project/condor-tensorflow/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/raschka-research-group/condor_tensorflow/blob/main/LICENSE)
![Python 3](https://img.shields.io/badge/python-3-blue.svg)

<br>

---

## About  

CONDOR, short for CONDitional Ordinal Regression, is a method for ordinal regression with deep neural networks, 
which addresses the rank inconsistency issue of other ordinal regression frameworks.

It is compatible with any state-of-the-art deep neural network architecture,
requiring only modification of the output layer, the labels, the loss function.

We also have condor [implemented for pytorch](https://github.com/GarrettJenkinson/condor_pytorch).

This package includes:

  * Ordinal tensorflow loss function: `CondorOrdinalCrossEntropy`
  * Ordinal tensorflow error metric: `OrdinalMeanAbsoluteError`
  * Ordinal tensorflow error metric: `OrdinalEarthMoversDistance`
  * Ordinal tensorflow sparse loss function: `CondorSparseOrdinalCrossEntropy`
  * Ordinal tensorflow sparse error metric: `SparseOrdinalMeanAbsoluteError`
  * Ordinal tensorflow sparse error metric: `SparseOrdinalEarthMoversDistance`
  * Ordinal tensorflow activation function: `ordinal_softmax`
  * Ordinal sklearn label encoder: `CondorOrdinalEncoder`

<img src="./img/rankconsistent.png" width=500>

---

## Cite as
If you use CONDOR as part of your workflow in a scientific publication, please consider citing the CONDOR repository with the following DOI:

- [Jenkinson, Khezeli, Oliver, Kalantari, Klee. Universally rank consistent ordinal regression in neural networks, arXiv:2110.07470, 2021.](https://arxiv.org/abs/2110.07470)

```
@article{condor2021,
title = "Universally rank consistent ordinal regression in neural networks",
journal = "arXiv",
volume = "2110.07470",
year = "2021",
url = "https://arxiv.org/abs/2110.07470",
author = "Garrett Jenkinson and Kia Khezeli and Gavin R. Oliver and John Kalantari and Eric W. Klee",
keywords = "Deep learning, Ordinal regression, neural networks, Machine learning, Biometrics"
}
```
