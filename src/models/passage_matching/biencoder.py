import collections
from ctypes import Union
import logging
import random
from typing import Tuple, List

import numpy as np
import os
import torch
import torch.nn as nn
from torch import Tensor as T
import torch.nn.functional as F

from .model_utils import CheckpointState
from .data_utils import Tensorizer
from .biencoder_data import BiEncoderSample

# from .modeling import BertTensorizer


BiEncoderBatch = collections.namedtuple(
    "BiEncoderInput",
    [
        "query_ids",
        "query_segments",
        "passage_ids",
        "passage_segments",
        "labels",
        "encoder_type"
    ]
)


def dot_product_scores(q_vectors: T, p_vectors: T):
    """
    calculates the dot product scores between query vectors and passage vectors
    :param q_vectors: batch_size x hidden_size
    :param p_vectors: batch_size x hidden_size
    :return result: batch_size x batch_size
    """
    result = torch.matmul(q_vectors, torch.transpose(p_vectors, 0, 1))
    return result


class BiEncoder(nn.Module):
    """
    Bi-Encoder model component
    Adopts 2 BERT models as two seperate encoders for query and passage encoding.
    """

    def __init__(
            self,
            query_model: nn.Module,
            passage_model: nn.Module,
            fix_q_encoder: bool = False,
            fix_p_encoder: bool = False,
    ):
        super(BiEncoder, self).__init__()
        self.query_model = query_model
        self.passage_model = passage_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_p_encoder = fix_p_encoder

    @staticmethod
    def get_representation(
            sub_model: nn.Module,
            input_ids: T,
            input_segments: T,
            attention_mask: T,
            fix_encoder: bool = False,
            representation_token_pos: int = 0,
    ) -> Tuple[T, T, T]:

        sequence_output = None
        pooled_output = None
        hidden_states = None

        if input_ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    output = sub_model(
                        input_ids,
                        input_segments,
                        attention_mask,
                        representation_token_pos
                    )
                    sequence_output = output.last_hidden_state
                    pooled_output = output.pooler_output
                    hidden_states = output.hidden_states
                if sub_model.training:
                    sequence_output.requires_grad_(requires_grads=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:

                output = sub_model(
                    input_ids,
                    input_segments,
                    attention_mask,
                    representation_token_pos
                )
                sequence_output = output[0]
                pooled_output = output[1]
                hidden_states = output[2]

            return sequence_output, pooled_output, hidden_states

    def forward(
            self,
            query_ids: T,
            query_segments: T,
            query_attention_mask: T,
            passage_ids: T,
            passage_segments: T,
            passage_attention_mask: T,
            encoder_type: str = None,
            representation_token_pos: int = 0,
    ) -> Tuple[T, T]:

        q_encoder = self.query_model if encoder_type is None or encoder_type == "query" else self.passage_model
        _q_seq, q_pooled_out, _q_hidden = self.get_representation(
            q_encoder,
            query_ids,
            query_segments,
            query_attention_mask,
            self.fix_q_encoder,
            representation_token_pos
        )

        p_encoder = self.passage_model if encoder_type is None or encoder_type == "passage" else self.query_model
        _p_seq, p_pooled_out, _p_hidden = self.get_representation(
            p_encoder,
            passage_ids,
            passage_segments,
            passage_attention_mask,
            self.fix_p_encoder,
            representation_token_pos
        )

        return q_pooled_out, p_pooled_out

    def create_biencoder_input(
            self,
            samples: List[BiEncoderSample],
            tensorizer: Tensorizer,
    ) -> BiEncoderBatch:

        query_tensors = []
        passage_tensors = []
        # label_tensors.size() = [batch_size * batch_size]
        label_tensors = torch.eye(len(samples)).view(-1, 1)

        for sample in samples:
            query_psg = sample.query_passage
            positive_psg = sample.positive_passages[np.random.choice(len(sample.positive_passages))]
            query_tensors.append(tensorizer.text_to_tensor(query_psg.abstract))
            passage_tensors.append(tensorizer.text_to_tensor(positive_psg.abstract))

        # passage_tensors.size() = [batch_size, max_seq_len]
        passage_tensors = torch.cat([psg.view(1, -1) for psg in passage_tensors], dim=0)
        # query_tensors.size() = [batch_size, max_seq_len]
        query_tensors = torch.cat([q.view(1, -1) for q in query_tensors], dim=0)

        passage_segments = torch.zeros_like(passage_tensors)
        query_segments = torch.zeros_like(query_tensors)

        return BiEncoderBatch(
            query_ids=query_tensors,
            query_segments=query_segments,
            passage_ids=passage_tensors,
            passage_segments=passage_segments,
            labels=label_tensors,
            encoder_type="query"
        )

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        self.load_state_dict(saved_state.model_dict, strict=strict)

    def get_state_dict(self):
        return self.state_dict()


class BiEncoderCELoss(object):

    @staticmethod
    def get_scores(q_vectors: T, p_vectors: T) -> T:
        return dot_product_scores(q_vectors, p_vectors)
        # return size = (num_q, num_p)

    def calculate(
            self,
            q_vectors: T,
            p_vectors: T,
            labels: T,
            loss_scale: float = None,
    ) -> Tuple[T, int]:
        # scores.size() = [batch_size, batch_size]
        scores = self.get_scores(q_vectors, p_vectors)
        softmax_scores = nn.functional.softmax(scores, dim=1)

        loss = nn.functional.binary_cross_entropy(softmax_scores.view(-1, 1), labels, reduction='mean')

        predictions: T = torch.argmax(softmax_scores, dim=1)
        correct_predictions_count = (predictions == torch.tensor(np.arange(scores.size(0))).to(predictions.device)).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count
