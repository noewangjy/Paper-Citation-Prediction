from typing import List, Dict
import json
import numpy as np


def summarize_authors(authors_path: str) -> List[str]:
    def split_fn(line: str) -> List[str]:
        line = line.replace('\n', '')
        _, authors = line.split('|--|')
        return authors.split(',')

    with open(authors_path, 'r') as f:
        all_authors = map(split_fn, f.readlines())

    res = set()
    for authors in all_authors:
        res.update(set(authors))

    return list(res)


if __name__ == '__main__':
    import sys
    AUTHORS_TXT_PATH: str = input("Input the path of authors.txt") if len(sys.argv) < 2 else sys.argv[1]
    AUTHORS_INDEX_PATH: str = input("Input the path of authors.json") if len(sys.argv) < 3 else sys.argv[2]
    author_list: List[str] = summarize_authors(AUTHORS_TXT_PATH)
    author_list.sort()
    values = np.arange(len(author_list)) / len(author_list) + 1 / len(author_list)

    author_index: Dict[str, float] = {}
    for idx, item in enumerate(author_list):
        author_index[item] = values[idx]

    with open(AUTHORS_INDEX_PATH, 'w') as f:
        json.dump(author_index, f, indent=4)
    print('finish')
