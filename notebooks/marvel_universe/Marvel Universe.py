#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import urllib.request as request
from SPARQLWrapper import SPARQLWrapper, JSON


# ## Creating the Hero - Comic Target Graph
# 
# The hero-comic book dataset on github contains the name of each Marvel hero, the comics that they have appeared in and the birth name of the hero (if available).
# 
# Let's download the dataset and inspect it. 
# The hero column can contain two values seperated by a slash so we'll split these out into seperate columns.

# In[ ]:


def get_id(url):
    """ A function to map the python hash function onto 32-bit integers"""
    return hash(url) % 2147483647


# In[ ]:


fp = request.urlopen('http://syntagmatic.github.io/exposedata/marvel/data/source.csv')
hero_comic_name_df = pd.read_csv(fp, names=['hero', 'comic'])
hero_comic_name_df['hero_id'] = list(map(get_id, hero_comic_name_df['hero'].str.upper()))
hero_comic_name_df['hero'] = hero_comic_name_df.hero.str.split('|').str.get(0).str.strip()
hero_comic_name_df['name'] = hero_comic_name_df.hero.str.split('/').str.get(1).str.strip()
hero_comic_name_df['hero'] = hero_comic_name_df.hero.str.split('/').str.get(0).str.strip()
hero_comic_name_df['name_id'] = list(map(get_id, hero_comic_name_df['name'].str.lower()))
hero_comic_name_df['comic_id'] = list(map(get_id, hero_comic_name_df['comic'].str.lower()))
hero_comic_name_df.head()


# Really the dataset ought to be represented by two tables. 
# One representing heros and their names.
# The other representing heros and the comics they have featured in.
# Let's do that to remove duplicates and NaN entries.

# In[ ]:


hero_name_df = hero_comic_name_df[['hero_id', 'name_id', 'hero', 'name']].drop_duplicates()
hero_name_df['name'].replace('', np.nan, inplace=True)
hero_name_df = hero_name_df[hero_name_df['name'].notnull()]
hero_name_df.head()


# In[ ]:


hero_comic_df = hero_comic_name_df[['hero_id', 'comic_id', 'hero', 'comic']].drop_duplicates()
hero_comic_df.head()


# Finally we could like to represent heros, names and comics with unique 32 bit integer identifiers rather than strings.
# We'll do this using the function `get_id` that maps the python hash funtion onto the range of 32-bit ints.
# This allows us to make one table of nodes and one table of edges which is a more ideomatic representation of a graph.

# In[ ]:


heros = hero_comic_df[['hero_id', 'hero']].drop_duplicates()
comic = hero_comic_df[['comic_id', 'comic']].drop_duplicates()
names = hero_name_df[['name_id', 'name']].drop_duplicates()

nodes = pd.concat(
    [
        pd.DataFrame({
            'id': heros['hero_id'],
            'label': heros['hero'],
            'type': 0
        }),
        pd.DataFrame({
            'id': comic['comic_id'],
            'label': comic['comic'],
            'type': 1
        }),
        pd.DataFrame({
            'id': names['name_id'],
            'label': names['name'],
            'type': 2
        }),  
    ]
)


# In[ ]:


edges = pd.concat(
    [
        pd.DataFrame({
            'start': hero_comic_df['hero_id'],
            'end': hero_comic_df['comic_id'],
        }),
        pd.DataFrame({
            'start': hero_name_df['hero_id'],
            'end': hero_name_df['name_id'],
        }),
        pd.DataFrame({
            'end': hero_comic_df['hero_id'],
            'start': hero_comic_df['comic_id'],
        }),
        pd.DataFrame({
            'end': hero_name_df['hero_id'],
            'start': hero_name_df['name_id'],
        }),
    ],
    sort=False
)


# In[ ]:


len(nodes), len(edges)


# In[ ]:


nodes.head()


# In[ ]:


edges.head()


# In[ ]:


nodes.to_csv('./target_nodes.csv', index=False)
edges.to_csv('./target_edges.csv', index=False)


# # Creating the Hero - Team Query Graph
# 
# The query graph is created using a SPARQL query for Marvel Heros, Aliases and Groups.

# In[ ]:


sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setQuery("""
    SELECT ?character ?characterLabel ?group ?groupLabel ?birthName ?characterAltLabel 
    WHERE {
        ?group wdt:P31 wd:Q14514600 ;  # group of fictional characters
              wdt:P1080 wd:Q931597.    # from Marvel universe
        ?character wdt:P463 ?group.    # member of group
        optional{ ?character wdt:P1477 ?birthName. }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en".}
    }
""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()


# In[ ]:


# load the results into a pandas DataFrame
records = []
for result in results["results"]["bindings"]:
    character_id = result['character']['value']
    group_id = result['group']['value']
    name = result['characterLabel']['value']
    group = result['groupLabel']['value']
    alt_names = None
    if 'characterAltLabel' in result:
        alt_names = result['characterAltLabel']['value']
    birth_name = None
    if 'birthName' in result:
        birth_name = result['birthName']['value']
    records.append((character_id, group_id, name, group, birth_name, alt_names))

frame = pd.DataFrame.from_records(records, columns=['character_id', 'group_id', 'name', 'group', 'birth_name', 'alt_names'])


# In[ ]:


names = frame[['character_id', 'name']].drop_duplicates()
groups = frame[['group_id', 'group']].drop_duplicates()
character_group = frame[['character_id', 'group_id']].drop_duplicates()
birth_names = frame[
    frame['birth_name'].notna() # do not include a row for characters without a birthname
][['character_id', 'birth_name']].drop_duplicates()


# In[ ]:


records = []
for uid, alt_names in zip(frame['character_id'], frame['alt_names']):
    if alt_names is None:
        continue
    for name in alt_names.split(','):
        records.append({'character_id': uid, 'alt_name': name})
alt_names = pd.DataFrame.from_records(records).drop_duplicates()


# In[ ]:


nodes = pd.concat(
    [
        pd.DataFrame({
            'id': list(map(get_id, names['character_id'])), 
            'label': names['name'],
            'type': 0
        }),
        pd.DataFrame({
            'id': list(map(get_id, groups['group_id'])), 
            'label': groups['group'],
            'type': 1, 
        }),
        pd.DataFrame({
            'id': list(map(get_id, birth_names['birth_name'])), 
            'label': birth_names['birth_name'].str.strip(),
            'type': 2, 
        }),
        pd.DataFrame({
            'id': list(map(get_id, alt_names['alt_name'])), 
            'label': alt_names['alt_name'].str.strip(),
            'type': 2, 
        })
    ], 
    sort=True
).drop_duplicates()
nodes.to_csv('./query_nodes.csv', index=False)


# In[ ]:


edges = pd.concat([
    # character to group
    pd.DataFrame([
        {'start': get_id(start), 'end': get_id(end)}
        for start, end in zip(character_group['character_id'], character_group['group_id'])
    ]),
#     # group to character
    pd.DataFrame([
        {'start': get_id(end), 'end': get_id(start)}
        for start, end in zip(character_group['character_id'], character_group['group_id'])
    ]),
    # character to alt name
    pd.DataFrame([
        {'start': get_id(start), 'end': get_id(end)}
        for start, end in zip(alt_names['character_id'], alt_names['alt_name'])
    ]),
    # alt name to character
    pd.DataFrame([
        {'start': get_id(end), 'end': get_id(start)}
        for start, end in zip(alt_names['character_id'], alt_names['alt_name'])
    ])
])
edges.to_csv('./query_edges.csv', index=False)


# ## Joining the Datasets

# In[2]:


import fornax
from sqlalchemy import create_engine
from sqlalchemy.orm.session import Session
from sqlalchemy.orm import Query
from sqlalchemy import literal


# In[3]:


engine = create_engine('sqlite://', echo=False)
connection = engine.connect()
fornax.model.Base.metadata.create_all(connection)

target_nodes_df = pd.read_csv('./target_nodes.csv')
# create a list of TargetNode objects
target_nodes = [
    fornax.model.TargetNode(id=uid, type=type_) 
    for uid, type_ in zip(target_nodes_df['id'], target_nodes_df['type'])
]

session = Session(connection)
session.add_all(target_nodes)
session.commit()

target_edges_df = pd.read_csv('./target_edges.csv')
# create a list of TargetEdge objects
target_edges = [fornax.model.TargetEdge(start=start, end=end) for start, end in zip(target_edges_df['start'], target_edges_df['end'])]


session.add_all(target_edges)
session.commit()

query_nodes_df = pd.read_csv('./query_nodes.csv')
query_edges_df = pd.read_csv('./query_edges.csv')
query_nodes = [
    fornax.model.QueryNode(id=uid, type=type_) for uid, type_ in zip(query_nodes_df['id'], query_nodes_df['type'])
]


session.add_all(query_nodes)
session.commit()

# create a list of TargetEdge objects
query_edges = [fornax.model.QueryEdge(start=start, end=end) for start, end in zip(query_edges_df['start'], query_edges_df['end'])]

session.add_all(query_edges)
session.commit()


# In[4]:


seed = Query([
    fornax.model.QueryNode.id.label('neighbour'),
    literal(0).label('distance')
]).filter(fornax.model.QueryNode.id == 1241907327)
query = fornax.select.neighbours(fornax.model.QueryNode, seed, 1)


# In[5]:


query_node_ids, distances = zip(*query.with_session(session).all())
query_node_ids = list(query_node_ids)


# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import BallTree

count_vectorizer = CountVectorizer(analyzer='char_wb', lowercase=True, ngram_range=[3, 3])
search_tree = BallTree(count_vectorizer.fit_transform(target_nodes_df['label'].str.lower()).toarray(), metric='jaccard')


# In[7]:


neighbours = query_nodes_df.set_index('id').loc[query_node_ids]
matches_array, all_distances = search_tree.query_radius(
    count_vectorizer.transform(neighbours['label'].str.lower()).toarray(), 
    r=.7,
    return_distance=True
)


# In[8]:


matches = []
for query_node_id, target_node_offsets, distances in zip(query_node_ids, matches_array, all_distances):
    for target_node_offset, distance in zip(target_node_offsets, distances):
        
        matches.append(
            fornax.model.Match(
                start=int(query_node_id), 
                end=int(target_nodes_df.iloc[target_node_offset]['id']), 
                weight=1. - distance
            )
        )


# In[9]:


session.add_all(matches)
session.commit()


# In[10]:


match_starts = set(m.start for m in matches)
match_ends = set(m.end for m in matches)

for q in query_edges:
    if q.start in match_starts and q.end in match_starts:
        continue
    session.delete(q)
        

for q in query_nodes:
    if q.id in match_starts:
        continue
    session.delete(q)
session.commit()


# In[11]:


get_ipython().run_cell_magic('time', '', 'batched_records, i, batch_size, finished = [], 0, 10000, False\nwhile not finished:\n    query = fornax.select.join(2, [i, i+batch_size])\n    next_batch = query.with_session(session).all()\n    batched_records += next_batch\n\n    if len(next_batch) == 0:\n        finished = True\n\n    i += batch_size')


# In[12]:


get_ipython().run_line_magic('time', 'subs = fornax.opt.solve(batched_records, max_iters=20, n=3)')


# In[13]:


for q, t in subs[0][0]:
    print(
        query_nodes_df[query_nodes_df['id'] == q]['label'].iloc[0] 
        + ' <--> ' 
        + target_nodes_df[target_nodes_df['id'] == t]['label'].iloc[0]
    )


# In[14]:


subs[0]


# In[ ]:


subs[0][1]


# In[ ]:


ids = pd.read_sql("""SELECT match.end FROM match""", con=connection)['end']
target_nodes_df.set_index('id').loc[ids]['label'].unique()


# In[ ]:


ids = pd.read_sql("""SELECT match.start FROM match""", con=connection)['start']
query_nodes_df.set_index('id').loc[ids]['label'].unique()


# In[ ]:


target_nodes_df.iloc[[490, 962, 2870, 3734, 4035, 5822, 6214, 11576, 15693]]


# In[ ]:


import pickle
with open('../../records.pk', 'wb') as fp:
    pickle.dump(batched_records, fp)


# In[ ]:





# In[ ]:


query_nodes_df[query_nodes_df['id'] == 952046635]


# In[ ]:


target_nodes_df[target_nodes_df['id'] == 1432924047]


# In[ ]:


sorted(query_nodes_df['id'])


# In[ ]:


pd.read_sql("""SELECT * FROM query_node WHERE id = 1421458225""", con=connection)


# In[ ]:


query_nodes_df[query_nodes_df['label'] == 'Avengers']


# In[ ]:


pd.read_sql("""SELECT * FROM query_node WHERE id = 1432924047""", con=connection)


# In[ ]:


pd.read_sql("""SELECT * FROM target_node WHERE id = 1432924047""", con=connection)


# In[ ]:


pd.read_sql("""SELECT * FROM query_node LIMIT 10""", con=connection)


# In[ ]:


query_nodes_df[query_nodes_df['id'] == 952046635]


# In[ ]:


target_nodes_df[target_nodes_df['id'] == 1605271291]


# In[ ]:




