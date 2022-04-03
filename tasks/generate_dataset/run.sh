export PYTHONPATH=$PYTHONPATH:../../
python generate_author_index.py ../../data/authors.txt ../../data/authors.json
python generate_dataset.py ../../data ../../data/neo_converted