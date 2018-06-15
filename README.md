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
