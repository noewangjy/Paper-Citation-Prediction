from torch import Tensor as T
import json
import logging
import pickle
import jsonlines
from typing import List, Iterator, Callable, Tuple

logger = logging.getLogger()


def read_serialized_data_from_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "rb") as reader:
            logger.info("Reading file %s", path)
            data = pickle.load(reader)
            results.extend(data)
            logger.info("Aggregated data size: {}".format(len(results)))
    logger.info("Total data size: {}".format(len(results)))
    return results


def read_data_from_json_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            logger.info("Reading file %s" % path)
            data = json.load(f)
            results.extend(data)
            logger.info("Aggregated data size: {}".format(len(results)))
    return results


def read_data_from_jsonl_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        logger.info("Reading file %s" % path)
        with jsonlines.open(path, mode="r") as jsonl_reader:
            data = [r for r in jsonl_reader]
            results.extend(data)
            logger.info("Aggregated data size: {}".format(len(results)))
    return results


def normalize_question(question: str) -> str:
    question = question.replace("’", "'")
    return question


class Tensorizer(object):
    """
    Component for all text to model input data conversion
    and related utility methods
    """

    def text_to_tensor(
            self,
            text: str,
            add_special_tokens: bool = True,
            apply_max_len: bool = True
    ):
        raise NotADirectoryError

    def get_pad_id(self) -> int:
        raise NotImplementedError

    def get_attention_mask(self, tokens_tensor: T):
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        raise NotImplementedError

    def to_string(self, token_ids, skip_special_tokens=True):
        raise NotImplementedError

    def set_pad_to_max(self, pad: bool):
        raise NotImplementedError

    def get_token_id(self, token: str) -> int:
        raise NotImplementedError


class RepTokenSelector(object):
    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        raise NotImplementedError


class RepStaticPosTokenSelector(RepTokenSelector):
    def __init__(self, static_position: int = 0):
        self.static_position = static_position

    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        return self.static_position


DEFAULT_SELECTOR = RepStaticPosTokenSelector()















