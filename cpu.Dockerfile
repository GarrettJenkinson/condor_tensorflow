# modified from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/cpu-jupyter.Dockerfile

ARG UBUNTU_VERSION=18.04

FROM ubuntu:${UBUNTU_VERSION} as base

RUN apt-get update && apt-get install -y curl

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade \
    "pip<20.3" \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.
ARG TF_PACKAGE=tensorflow
ARG TF_PACKAGE_VERSION=2.4.1
RUN python3 -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir tensorflow-hub jupyter matplotlib numpy==1.19.5 pandas scikit-learn==0.24.2
# Pin ipykernel and nbformat; see https://github.com/ipython/ipykernel/issues/422
RUN python3 -m pip install --no-cache-dir jupyter_http_over_ws ipykernel==5.1.1 nbformat==4.4.0
RUN jupyter serverextension enable --py jupyter_http_over_ws

COPY ./ /etc/condor_tensorflow/
RUN chmod -R +rwx /etc/condor_tensorflow
RUN python3 -m pip install /etc/condor_tensorflow

RUN mkdir -p /condor/condor-tutorials && chmod -R a+rwx /condor/
COPY ./docs/CONDOR_TensorFlow_demo.ipynb /condor/condor-tutorials/

RUN mkdir /.local && chmod a+rwx /.local
RUN apt-get update && apt-get install -y --no-install-recommends wget git
RUN apt-get autoremove -y && apt-get remove -y wget
WORKDIR /condor
EXPOSE 8888

RUN python3 -m ipykernel.kernelspec

CMD ["bash", "-c", "source /etc/condor_tensorflow/docker_bashrc && jupyter notebook --notebook-dir=/condor --ip 0.0.0.0 --no-browser --allow-root"]

