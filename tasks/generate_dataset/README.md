# Generate Dataset

This task aims to generate a dataset from:

- `edgelist.txt`
- `authors.txt`
- `abstracts.txt`
- `test.txt`

## Intro

There is this `config.yaml` under `conf` directory:
```yaml
features:
  pos_uv:
    deps: [ ]
    products:
      - pos_u
      - pos_v
      - u
      - v
      - y
  neg_uv:
    deps: [ ]
    products:
      - neg_u
      - neg_v
      - u
      - v
      - y
  edges:
    deps: [ ]
    products:
      - origin_edges
  graph:
    deps: [ ]
    products:
      - graph
  node_abstract_bert:
    deps: [ ]
    products:
      - node_abstract_bert
  edge_generic:
    deps:
      - u
      - v
      - y
    products:
      - edge_generic
  cora_like:
    deps: [ ]
    products:
      - cora_features
      - cora_vocab
      - cora_word_index
  authors_index:
    deps: [ ]
    products:
      - authors_index

...
```

The names under `features` tag are available tabes. `feature.products` are names they will produce in final datset

You can combine multiple features in the target_features section.


## Usage

### Generate

To use the `no_feature` profile under `conf/target_features` directory:
```bash
python generate_dataset target_features=no_feature prefix=nullptr_no_feature
```

where `src` stores the `txt` files and `dst` contains output. The output is `nullptr_no_feature_[train|dev].[pkl|npz]`

### Load

The converted dataset is a collective of numpy arrays

```python
import pickle

with open('nullptr_no_feature_train.pkl', 'rb') as f:
    data = pickle.load(f)
```



The following keys can be used to index the converted dataset

- `abstracts` Array of shape`[N, ]`, dtype=`str`
- `authors` Array of shape`[N, ]`, dtype=`List[str]`
- `origin_edges` Array of shape`[M, 2]`, dtype=`int`
- `pos_u` Array of shape`[M',]`, dtype=`int`
- `pos_v` Array of shape`[M',]`, dtype=`int`
- `neg_u` Array of shape`[M',]`, dtype=`int`
- `neg_v` Array of shape`[M',]`, dtype=`int`

`M` is the number of edges in original graph, `N` is the number of nods and `M'` the number of edges in subset

Other profiles will generate a datset with additional keys