# Feature Extraction

This document introduces how we obtained features from original dataset.

## Get started

Put `edgelist.txt`, `authors.txt`, `test.txt` and `abstracts.txt`  under the `{PROJECT_ROOT}/data` directory

```text
.
├── abstracts.txt
├── authors.json
├── authors.txt
└──  test.txt
```

## Generate datasets

Navigate to `{PROJECT_ROOT}/tasks/generate_dataset`. Execute the following command in shell

```shell
export PYTHONPATH=$PYTHONPATH:../../
python generate_dataset.py target_features=no_feature prefix=nullptr_no_feature output_path=../../data/neo_converted
python generate_dataset.py target_features=graph_conv prefix=nullptr_graph_conv output_path=../../data/converted
```

By doing this the script will dump 8 serialized dataset under `{PROJECT_ROOT}/data/converted` and `{PROJECT_ROOT}/tasks/neo_converted`.

**nullptr_no_feature**

This dataset containes positve edges from `edgelist.txt`, generated negative edges. It also split authors and abstract to lists. (Basic information)

**nullptr_graph_conv**

This dataset contains basic information. It constructs the author network using reference links and co-author relationship. It also generates a cora-like feature, which is the frequency count of frequent words.

> This may take many minutes, even hours.
> This is selective if you only want to use enhanced baseline

Make sure the presence of these files. Then execute this command to generate a list of $U$, $V$ pairs

```shell
python generate_uv_list.py
```

By doing this the script will dump `uv_list.pkl` and `uv_list.chksum` (The md5sum of `uv_list.pkl`) under `{PROJECT_ROOT}/data/converted` and `{PROJECT_ROOT}/tasks/neo_converted`. The `uv_list` contains the index of vertices together with labels of train dataset, dev dataset and test dataset.

> It is very important that a globally unique uv_list is used to fuse different features.

## Generate features

We have to generate 3 features and store them in a serialized pickle

| Name                           | Content                                                  |
| ------------------------------ | -------------------------------------------------------- |
| `baseline-enhanced.pkl`        | graph / text based features extracted from essay network |
| `author_graph_lr_features.pkl` | graph based features extracted from author network       |
| `graphsage_essay_features.pkl` | features extracted by GraphSAGE model on essay network   |

### baseline-enhanced.pkl

Navigate to `{PROJECT_ROOT}/tasks/run_new_baseline/`, then execute the `run.sh`

> This may take many minutes, even hours.

### author_graph_lr_features.pkl

Navigate to `{PROJECT_ROOT}/tasks/run_authorgraph_lr/`, then execute the `run.sh`

This will dump the `author_graph_lr_features.pkl` under `{PROJECT_ROOT}/tasks/run_authorgraph_lr/`.

### graphsage_essay_features.pkl

Navigate to `{PROJECT_ROOT}/tasks/train_graphsage_essay/`, then execute the `run.sh`

This will launch training of GraphSAGE model on essay network. After the training, exam the logs and pick a proper checkpoint(hidden state) under the `{PROJECT_ROOT}/tasks/train_graphsage_essay/ouputs/{YYYY-MM-DD}/{HH-mm-SS}`. Its name starts with *epoch* which is the number of epochs. Remember its path.

In the same directory, run `digest.py` with path to checkpoint as the first argument:

```shell
python digest.py {PATH_TO_CHECKPOPINT}
```

This will dump the `graphsage_essay_features.pkl` under current directory.

> This may take many minutes, even hours.
