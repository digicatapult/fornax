[![CircleCI](https://circleci.com/gh/CDECatapult/fornax.svg?style=svg&circle-token=2110b6bc1d713698d241fd08ae60cd925e60062f)](https://circleci.com/gh/CDECatapult/fornax)

# Fornax

An implementation of [NeMa: Fast Graph Search with Label Similarity](http://www.vldb.org/pvldb/vol6/p181-khan.pdf) using a sqlite or postgres backend with python3.

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
query_graph_handle = fornax.GraphHandle.create()
query_graph_handle.add_nodes(label=['Hulk', 'Lady', 'Storm'])
query_graph_handle.add_edges([0, 1], [1, 2])


# create a target graph
target_graph_handle = fornax.GraphHandle.create()
target_graph_handle.add_nodes(label=comic_book_nodes['name'])
target_graph_handle.add_edges(comic_book_edges['start'], comic_book_edges['end'])

matches = [
    (query_node_id, target_node_id, weight) 
    for query_node_id, target_node_id, weight 
    in string_similarities
]

match_starts, match_ends, weights = zip(*matches)

# stage a query
query = fornax.QueryHandle.create(query_graph_handle, target_graph_handle)
query.add_matches(match_starts, match_ends, weights)

# go!
query.execute()
```

```json
{
    "graphs": [
        {
            "cost": 0.024416640711327393,
            "nodes": [
                {
                    "id": 5174074480569935113,
                    "type": "query",
                    "name": "hulk"
                },
                {
                    "id": 5174075668124758188,
                    "type": "query",
                    "name": "lady"
                },
                {
                    "id": 5174076855952377563,
                    "type": "query",
                    "name": "storm"
                },
                {
                    "id": -1950980926759484095,
                    "type": "target",
                    "uid": 2142361735,
                    "label": "She-Hulk",
                    "type_": 0
                },
                {
                    "id": -1951729878043816045,
                    "type": "target",
                    "uid": 995920086,
                    "label": "Lady Liberators",
                    "type_": 1
                },
                {
                    "id": -1951205851414851420,
                    "type": "target",
                    "uid": 37644418,
                    "label": " Susan Storm",
                    "type_": 2
                }
            ],
            "links": [
                {
                    "start": 5174074480569935113,
                    "end": -1950980926759484095,
                    "type": "match",
                    "weight": 0.9869624795392156
                },
                {
                    "start": 5174075668124758188,
                    "end": -1951729878043816045,
                    "type": "match",
                    "weight": 0.9746778514236212
                },
                {
                    "start": 5174076855952377563,
                    "end": -1951205851414851420,
                    "type": "match",
                    "weight": 0.9651097469031811
                },
                {
                    "start": 5174074480569935113,
                    "end": 5174075668124758188,
                    "type": "query",
                    "weight": 1.0
                },
                {
                    "start": 5174075668124758188,
                    "end": 5174076855952377563,
                    "type": "query",
                    "weight": 1.0
                }
            ]
        }
    ],
    "iters": 2,
    "hopping_distance": 2,
    "max_iters": 10
}
```

## Tutorials

See the tutorials for a full working example

* [Part 1](https://github.com/CDECatapult/fornax/blob/master/notebooks/tutorial/Tutorial%201%20-%20Creating%20a%20Dataset.ipynb)
* [Part 2](https://github.com/CDECatapult/fornax/blob/master/notebooks/tutorial/Tutorial%202%20-%20Making%20a%20Query.ipynb)
