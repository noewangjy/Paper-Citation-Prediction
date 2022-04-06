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
        "query_passage_ids",
        "query_passage_segments",
        "passage_ids",
        "passage_segments",
        "positive_passage_indices",
        "hard_negative_passage_indices",
        "encoder_type"
    ]
)


def dot_product_scores(q_vectors: T, p_vectors: T):
    """
    calculates the dot product scores between query vectors and passage vectors
    :param q_vectors: n1 x D
    :param p_vectors: n2 x D
    :return result: n1 x n2
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
            num_negatives: int = 0,
            num_hard_negatives: int = 0,
            shuffle: bool = False,
            shuffle_positives: bool = False,
    ) -> BiEncoderBatch:

        query_tensors = []
        passage_tensors = []
        positive_passage_indices = []
        hard_negative_passage_indices = []

        for sample in samples:

            # Get 1 positive passage randomly or certainly
            if shuffle and shuffle_positives:
                positive_psgs = sample.positive_passages
                positive_psg = positive_psgs[np.random.choice(len(positive_psgs))]
            else:
                positive_psg = sample.positive_passages[0]

            negative_psgs = sample.negative_passages
            hard_neg_psgs = sample.hard_negative_passages
            query_passage = sample.query_passage

            if shuffle:
                random.shuffle(negative_psgs)
                random.shuffle(hard_neg_psgs)

            negative_psgs = negative_psgs[:num_negatives]
            hard_neg_psgs = hard_neg_psgs[:num_hard_negatives]

            all_psgs = [positive_psg] + negative_psgs + hard_neg_psgs
            hard_neg_start_idx = 1
            hard_neg_end_idx = 1 + len(hard_neg_psgs)

            current_psgs_len = len(passage_tensors)

            sample_psg_tensors = [tensorizer.text_to_tensor(psg.abstract) for psg in all_psgs]

            passage_tensors.extend(sample_psg_tensors)
            positive_passage_indices.append(current_psgs_len)
            hard_negative_passage_indices.append(
                list(range(current_psgs_len + hard_neg_start_idx, current_psgs_len + hard_neg_end_idx))
            )

            query_tensors.append(tensorizer.text_to_tensor(query_passage.abstract))

        passage_tensors = torch.cat([psg.view(1, -1) for psg in passage_tensors], dim=0)
        query_tensors = torch.cat([q.view(1, -1) for q in query_tensors], dim=0)

        passage_segments = torch.zeros_like(passage_tensors)
        query_segments = torch.zeros_like(query_tensors)

        return BiEncoderBatch(
            query_passage_ids=query_tensors,
            query_passage_segments=query_segments,
            passage_ids=passage_tensors,
            passage_segments=passage_segments,
            positive_passage_indices=positive_passage_indices,
            hard_negative_passage_indices=hard_negative_passage_indices,
            encoder_type="query"
        )

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        self.load_state_dict(saved_state.model_dict, strict=strict)

    def get_state_dict(self):
        return self.state_dict()


class BiEncoderNllLoss(object):

    @staticmethod
    def get_scores(q_vectors: T, p_vectors: T) -> T:
        return dot_product_scores(q_vectors, p_vectors)
        # return size = (num_q, num_p)

    def calculate(
            self,
            q_vectors: T,
            p_vectors: T,
            positive_indices: list,
            hard_negative_indices: list = None,
            loss_scale: float = None,
    ) -> Tuple[T, int]:

        scores = self.get_scores(q_vectors, p_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(softmax_scores,
                          torch.tensor(positive_indices).to(softmax_scores.device),
                          reduction="mean"
                          )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_indices).to(max_idxs.device)).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count














