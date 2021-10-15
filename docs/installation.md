# Installing `condor_tensorflow`


## Requirements

`condor_tensorflow` has been tested with the following software and packages:

- [Python](https://www.python.org) == 3.9.6
- [Tensorflow](http://www.tensorflow.org) == 2.4.1
- [sklearn](https://scikit-learn.org/) == 0.24.2 
- [numpy](https://numpy.org/) == 1.19.5 (newer versions currently incompatibile with above tensorflow version)

## PyPI

You can install the latest stable release of `condor_tensorflow` directly from Python's package index via `pip` by executing the following code from your command line:  

```bash
pip install condor-tensorflow
```


## Latest GitHub Source Code

<br>

You want to try out the latest features before they go live on PyPI? Install the `condor_tensorflow` dev-version latest development version from the GitHub repository by executing

```bash
pip install git+git://github.com/GarrettJenkinson/condor_tensorflow.git
```

## Docker

This package relies on Python 3.6+, Tensorflow 2.2+, sklearn, and numpy.
For convenience we provide a Dockerfile that will build a container with
`condor_tensorflow` as well as its dependencies. This can be used
as

```bash
# Create a docker image
docker build -t cpu_tensorflow -f cpu.Dockerfile ./

# run image to serve a jupyter notebook
docker run -it -p 8888:8888 --rm cpu_tensorflow

# how to run bash inside container (with python that will have deps)
docker run -u $(id -u):$(id -g) -it -p 8888:8888 --rm cpu_tensorflow bash
```

Assuming a GPU enabled machine with the NVIDIA drivers installed replace `cpu`
above with `gpu`.

