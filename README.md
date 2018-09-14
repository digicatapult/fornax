# Fornax

An implementation of [NeMa: Fast Graph Search with Label Similarity](http://www.vldb.org/pvldb/vol6/p181-khan.pdf) using a postgres backend and python3.

## Install

From the root directory:

```bash
pip install -r requirements.txt .
``` 

Initialise a new postgres backend:

```bash
docker-compose up
```

## Test

From the root directory

```bash
python3 -m unittest discover -v -s ./test -p "test_*.py"
```

## Tutorials

See the tutorials to learn how to use this repo

* [Part 1](https://github.com/CDECatapult/fornax/blob/master/notebooks/tutorial/Tutorial%201%20-%20Creating%20a%20Dataset.ipynb)
* [Part 2](https://github.com/CDECatapult/fornax/blob/master/notebooks/tutorial/Tutorial%202%20-%20Initialise%20the%20database%20.ipynb)
* [Part 3](https://github.com/CDECatapult/fornax/blob/master/notebooks/tutorial/Tutorial%203%20-%20Making%20a%20Query.ipynb)
