# Fornax

[![CircleCI](https://circleci.com/gh/digicatapult/fornax.svg?style=svg&circle-token=2110b6bc1d713698d241fd08ae60cd925e60062f)](https://circleci.com/gh/digicatapult/fornax)
[![Coverage Status](https://coveralls.io/repos/github/digicatapult/fornax/badge.svg?branch=master)](https://coveralls.io/github/digicatapult/fornax?branch=master)
[![Known Vulnerabilities](https://snyk.io/test/github/digicatapult/fornax/badge.svg)](https://snyk.io/test/github/digicatapult/fornax/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/fornax/badge/?version=latest)](https://fornax.readthedocs.io/en/latest/?badge=latest)

An implementation of [NeMa: Fast Graph Search with Label Similarity](http://www.vldb.org/pvldb/vol6/p181-khan.pdf) using python3 and sqlite or postgres.

<!-- do not use a relative path because it won't work on PyPI -->
![FORNAX](https://github.com/digicatapult/fornax/raw/master/docs/img/fornax.png)

## Install

Via pip:

```bash
pip install fornax
```

Note that Fornax requires numpy to be installed (part of the SciPy ecosystem)
which in turn has non python dependencies.
The available options for installing SciPy packages are listed [here](https://scipy.org/install.html).

We recomend installing fornax via anaconda

```bash
conda create --name myenv python=3.6
source activate myenv
pip install fornax
```

## Install (Dev)

From the root directory:

```bash
# install dev dependencies
pip install -r requirements/dev.txt

# install fornax
pip install -e .
```

## View the Docs

View the docs at: [fornax.readthedocs.io](http://fornax.readthedocs.io/)

## Test

From the root directory

```bash
python run_test.py
```

## Tutorials

See the tutorials for a full working example.

* [Part 1](docs/tutorial/tutorial1.ipynb) - Download a small graph dataset
* [Part 2](docs/tutorial/tutorial2.ipynb) - Search the dataset using fornax

### Install Tutorial Dependencies (using conda)

The following tutorials use jupyter notebooks to create a worked example.
We recommend you use the anaconda python distribution to run the notebooks.

```bash
conda env create -f environment.yml
```

### Run the Tutorials

```bash
source activate fornax_tutorial
cd docs/tutorial
jupyter-notebook
```

## Documentation

### Build the Docs

```bash
# install docs dependencies
pip install -r requirements/docs.txt
# install fornax
pip install .

# build
cd docs
make html
```

### View the Docs Locally

```bash
cd _build/html
python3 -m http.server
```

navigate to `0.0.0.0:8000` in your browser.
