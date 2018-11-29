[![CircleCI](https://circleci.com/gh/CDECatapult/fornax.svg?style=svg&circle-token=2110b6bc1d713698d241fd08ae60cd925e60062f)](https://circleci.com/gh/CDECatapult/fornax)
[![Coverage Status](https://coveralls.io/repos/github/CDECatapult/fornax/badge.svg?branch=master)](https://coveralls.io/github/CDECatapult/fornax?branch=master)
[![Known Vulnerabilities](https://snyk.io/test/github/CDECatapult/fornax/badge.svg)](https://snyk.io/test/github/CDECatapult/fornax/badge.svg)


# Fornax

An implementation of [NeMa: Fast Graph Search with Label Similarity](http://www.vldb.org/pvldb/vol6/p181-khan.pdf) using python3 and sqlite or postgres.

![FORNAX](./docs/img/fornax.png)

## Install (Dev)

From the root directory:

```bash
pip install -r requirements/dev.txt
``` 

## Test

From the root directory

```bash
python run_test.py
```

Fornax requires numpy to be installed (part of the SciPy ecosystem). 
The available options for installing SciPy packages are listed [here](https://scipy.org/install.html).

## Tutorials

See the tutorials for a full working example.

* [Part 1](https://github.com/CDECatapult/fornax/blob/master/notebooks/tutorial/Tutorial%201%20-%20Creating%20a%20Dataset.ipynb) - Download a small graph dataset 
* [Part 2](https://github.com/CDECatapult/fornax/blob/master/notebooks/tutorial/Tutorial%202%20-%20Making%20a%20Query.ipynb) - Search the dataset using fornax

### Tutorial Dependencies

The following tutorials use jupyter notebooks to create a worked example.
We recommend you use the anaconda python distribution to run the notebooks.

```bash
conda env create -f environment.yml
source activate fornax_tutorial
```

## Documentation

### Build the Docs

```bash
cd docs
make html
cd _build/html
```

### View the Docs Locally

```bash
python3 -m http.server
```
