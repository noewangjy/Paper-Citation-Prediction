import json
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Tuple

import numpy as np
import tqdm
import nltk
from nltk.corpus import stopwords

INTERPUNCTUATIONS = {',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%'}
STOPS = set(stopwords.words("english"))
PROHIBITED_WORDS: set = set(['-', '--', '‐'] + [str(i) for i in range(10000)])
PROHIBITED_CHARACTERS: set = INTERPUNCTUATIONS | \
                             set([chr(idx) for idx in range(0x7f, 0x4000)]) | \
                             {'`', "'", '"', '\\', '~', '_', '|', '/', '<', '>', '{', '}', '=', '+', '^', '’', '“', '”', '‘', }
MAX_FREQ = 1e8
MIN_FREQ = 500


def read_abstracts(abstracts_path: str) -> List[str]:
    def split_fn(line: str) -> str:
        line = line.replace('\n', '').lower()
        if line:
            index, abstract = line.split('|--|')
            return abstract
        else:
            return ''

    with open(abstracts_path, 'r') as f:
        lines = f.readlines()
        result = map(split_fn, lines)

    return list(result)


def tokenize_fn(idx,
                abstract_list: List[str],
                tokenizer,
                interpunctuations: set,
                stops: set,
                prohibited_characters: set,
                prohibited_words: set) -> Dict[str, int]:
    def judge(word: str,
              interpunctuations: set,
              stops,
              prohibited_characters,
              prohibited_words):
        return (word not in interpunctuations) and (word not in stops) and (not any([i in prohibited_characters for i in set(word)])) and (word not in prohibited_words)

    num_abstracts: int = len(abstract_list)
    res: Dict[str, int] = {}
    with tqdm.tqdm(range(num_abstracts)) as pbar:
        pbar.set_description(f"Idx: {idx}")
        for idx in range(num_abstracts):
            for word in tokenizer(abstract_list[idx]):
                if judge(word, interpunctuations, stops, prohibited_characters, prohibited_words):
                    if word in res.keys():
                        res[word] += 1
                    else:
                        res[word] = 1
            if idx % 1e2 == 0:
                pbar.update(1e2)

    return res


def generate_dict(abstracts: List[str], tokenizer) -> Dict[str, int]:
    num_abstracts: int = len(abstracts)
    num_procs: int = 24
    batch_size: int = math.ceil(num_abstracts / num_procs)

    with ProcessPoolExecutor(max_workers=num_procs) as pool:
        results = []
        for idx in range(num_procs):
            results.append(pool.submit(tokenize_fn, idx, abstracts[idx * batch_size: (idx + 1) * batch_size], tokenizer, INTERPUNCTUATIONS, STOPS, PROHIBITED_CHARACTERS, PROHIBITED_WORDS))
        pool.shutdown()

    frequency_dict_reduced = {}
    with tqdm.tqdm(range(len(results))) as pbar:
        pbar.set_description(f"Reducing")
        for frequency_dict in map(lambda x: x.result(), results):
            for key in frequency_dict:
                if key in frequency_dict_reduced:
                    frequency_dict_reduced[key] += frequency_dict[key]
                else:
                    frequency_dict_reduced[key] = frequency_dict[key]
            pbar.update()

    frequency_dict_reduced_filtered = {k: v for k, v in frequency_dict_reduced.items() if MAX_FREQ > v > MIN_FREQ}

    return frequency_dict_reduced_filtered


def generate_abstract_feature(abstracts: List[str],
                              frequency_dict: Dict[str, int],
                              tokenizer) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    features = np.zeros(shape=(len(abstracts), len(frequency_dict) + 1), dtype=np.uint8)
    print("Estimated feature shape: ", features.shape)
    # features[:, -1] = 1  # Last column is always 1
    word_index: List[str] = sorted(list(frequency_dict.keys()))
    vocab = {k: v for v, k in enumerate(word_index)}

    num_abstracts: int = len(abstracts)
    with tqdm.tqdm(range(num_abstracts)) as pbar:
        pbar.set_description("Creating abstract feature")
        for idx in range(num_abstracts):
            for word in tokenizer(abstracts[idx]):
                if word in vocab.keys():
                    features[idx][vocab[word]] = 1
            if idx % 1e2 == 0:
                pbar.update(1e2)
            if features[idx].sum() <= 0:
                features[idx][-1] = 1  # Last column is 1 -> No keyword

    return features, vocab, word_index


class CustomTokenizer:
    def __init__(self):
        self.regex_tokenizer = nltk.RegexpTokenizer(r'\w+')

    def __call__(self, string):
        return list(map(lambda x: nltk.PorterStemmer().stem(x), self.regex_tokenizer.tokenize(string)))


if __name__ == '__main__':

    DEBUG: bool = False
    regex_tokenizer = CustomTokenizer()
    ABSTRACTS_TXT_PATH: str = input("Input the path of abstracts.txt") if len(sys.argv) < 2 else sys.argv[1]
    ABSTRACTS_INDEX_PATH: str = input("Input the path of output") if len(sys.argv) < 3 else sys.argv[2]
    if not os.path.exists(ABSTRACTS_INDEX_PATH):
        os.makedirs(ABSTRACTS_INDEX_PATH)

    abstracts: List[str] = read_abstracts(ABSTRACTS_TXT_PATH)
    if DEBUG:
        abstracts_frequency = generate_dict(abstracts[:10000], regex_tokenizer)
        abstract_features, abstract_vocab, abstract_index = generate_abstract_feature(abstracts[:10000], abstracts_frequency, regex_tokenizer)
    else:
        abstracts_frequency = generate_dict(abstracts, regex_tokenizer)
        abstract_features, abstract_vocab, abstract_index = generate_abstract_feature(abstracts, abstracts_frequency, regex_tokenizer)

    with open(os.path.join(ABSTRACTS_INDEX_PATH, 'vocab.json'), 'w') as f:
        json.dump(abstract_vocab, f, indent=4)
    with open(os.path.join(ABSTRACTS_INDEX_PATH, 'frequency.json'), 'w') as f:
        json.dump(abstracts_frequency, f, indent=4)
    with open(os.path.join(ABSTRACTS_INDEX_PATH, 'index.json'), 'w') as f:
        json.dump(abstract_index, f, indent=4)
    with open(os.path.join(ABSTRACTS_INDEX_PATH, 'features.pkl'), 'wb') as f:
        pickle.dump(abstract_features, f)

    np.savetxt(os.path.join(ABSTRACTS_INDEX_PATH, 'features.csv'), abstract_features, delimiter=', ', fmt="%d")
    print('finish')
