[![CircleCI](https://circleci.com/gh/CDECatapult/fornax.svg?style=svg&circle-token=2110b6bc1d713698d241fd08ae60cd925e60062f)](https://circleci.com/gh/CDECatapult/fornax)

# Fornax

An implementation of [NeMa: Fast Graph Search with Label Similarity](http://www.vldb.org/pvldb/vol6/p181-khan.pdf) using a postgres backend and python3.

## Install

From the root directory:

```bash
pip install -r requirements.txt .
``` 

## Test

From the root directory

```bash
python run_test.py
```

## Quick start

```python
# create a query graph
query_graph = fornax.GraphHandle.create(
    [0, 1, 2], 
    [(0, 1), (1, 2)], 
    metadata=[{'label': 'Hulk'}, {'label': 'Lady'}, {'label': 'Storm'}]
)

# create a target graph
query_graph = fornax.GraphHandle.create(
    comic_book_nodes, 
    comic_book_edges, 
    metadata=node_metadata
)

# stage a query
query = fornax.QueryHandle.create(query_graph, target_graph, matches)

# go!
results = query.execute(n=10, edges=True)
```

## Tutorials

See the tutorials for a working example

* [Part 1](https://github.com/CDECatapult/fornax/blob/master/notebooks/tutorial/Tutorial%201%20-%20Creating%20a%20Dataset.ipynb)
* [Part 2](https://github.com/CDECatapult/fornax/blob/master/notebooks/tutorial/Tutorial%202%20-%20Making%20a%20Query.ipynb)
