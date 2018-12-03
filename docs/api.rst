.. module:: fornax.api

API
===

.. _fornax-api-introduction:

Introduction
------------

This part of the documentation covers the the interface for creating an searching graphs using the fornax package.
For the full documentation of the module api see :ref:`fornax-api-module`.


All of the functionality in :mod:`fornax` can be accessed via the follwoing three classes.

* :class:`Connection`
* :class:`GraphHandle`
* :class:`QueryHandle`

:class:`Connection` is used to manage a connection to a SQL database.
:class:`GraphHandle` and :class:`QueryHandle` are used to create, insert
update and delete graphs and queries.

Connection API
--------------------


Fornax stores and queries graphs using a database via a database connection.
:class:`Connection` manages the lifecycle of this database connection,
the creation of database schema (if required)
and any cleanup once the connection is closed.


.. autoclass:: Connection
    :members:
    :noindex:

Graph API
--------------------------------

Since Graphs are persisted in a database they are not represented
directly by any object.
Rather, graphs are accessed via a graph handle which permits the user
to manipulate graphs via a :class:`Connection` instance.

.. autoclass:: GraphHandle
    :members:
    :noindex:

Query API
------------------------------

Like Graphs, queries exist in a database and a accessed via a handle.
Queries are executed using the :func:`QueryHandle.execute` method.

A query brings together three important concenpts.

A **target graph** is the graph which is going to be searched.

A **query graph** is the subgraph that is being seached for in the target graph.

**matches** are label similarities between nodes in the query graph and target graph
with a weight where :math:`0 \lt weight \lt= 1`.
Users are free to caculate label similarity scores however they like.
Fornax only needs to know about non zero weights between matches.

Once a query has been created and executed it will return the *n* subgraphs in the
target graph which are most similar to the query graph based on the similarity score
between nodes and their surrounding neighbourhoods.

.. note::
    Nodes in the target graph will only be returned from a query if they have a 
    non zero similarity score to at least one node in the query graph.


.. autoclass:: QueryHandle
    :members:
    :noindex:

