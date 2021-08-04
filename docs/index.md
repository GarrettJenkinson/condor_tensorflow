
<img src="./img/condor-logo-alpha.png" width=300>

**CORAL implementation for ordinal regression with deep neural networks.**

[![PyPi version](https://pypip.in/v/condor_pytorch/badge.png)](https://pypi.org/project/condor_pytorch/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/raschka-research-group/condor_pytorch/blob/main/LICENSE)
![Python 3](https://img.shields.io/badge/python-3-blue.svg)

<br>

---

## About  

CORAL, short for COnsistent RAnk Logits, is a method for ordinal regression with deep neural networks, which addresses the rank inconsistency issue of other ordinal regression frameworks.

<img src="img/figure1.jpg" width=400>

Originally, developed this method in the context of age prediction from face images. Our approach was evaluated on several face image datasets for age prediction using ResNet-34, but it is compatible with other state-of-the-art deep neural networks.

This repository implements the CORAL functionality (neural network layer, loss function, and dataset utilities) for convenient use. Examples are provided via the "Tutorials" in the upper left menu bar.

If you are looking for the orginal implementation, training datasets, and training log files corresponding to the paper, you can find these here: [https://github.com/Raschka-research-group/condor-cnn](https://github.com/Raschka-research-group/condor-cnn).


---

## Cite as

If you use CONDOR as part of your workflow in a scientific publication, please consider citing the CONDOR repository with the following DOI:

- TBD publication

```
@article{condor2021,
title = "TBD",
journal = "TBD",
volume = "TBD",
pages = "TBD",
year = "TBD",
issn = "TBD",
doi = "TBD",
url = "TBD",
author = "Garrett Jenkinson",
keywords = "Deep learning, Ordinal regression, neural networks, Machine learning, Biometrics"
}
```

