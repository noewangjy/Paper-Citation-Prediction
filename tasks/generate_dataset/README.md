# Generate Dataset

This task aims to generate a dataset from:

- `edgelist.txt`
- `authors.txt`
- `abstracts.txt`
- `test.txt`

## Usage

### Generate

```bash
python generate_dataset <src> <dst>
```

where `src` stores the `txt` files and `dst` contains output. The output is `nullptr_[train|dev].[pkl|npz]`

### Load

The converted dataset is a collective of numpy arrays

```python
import pickle

with open('nullptr_train.pkl', 'rb') as f:
    data = pickle.load(f)
```

```python
import numpy as np

npzfile = np.load('nullptr_train.npz')
keys = npzfile.keys()
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
