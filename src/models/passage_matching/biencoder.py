import collections
from typing import Tuple, List, Dict
import torch
import torch.nn as nn
from torch import Tensor as T
import torch.nn.functional as F

from .modeling import BertEncoder
from .model_utils import CheckpointState
from .data_utils import Tensorizer


# BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["authors", "abstract"])
from transformers import AutoModel, AutoConfig, AutoTokenizer

BiEncoderBatch = collections.namedtuple(
    "BiEncoderInput",
    [
        "query_ids",
        "query_segments",
        "context_ids",
        "context_segments",
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


class AutoBiEncoderCat(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model_name_or_path)
        self.query_model: nn.Module = AutoModel.from_pretrained(cfg.model_name_or_path, config = self.config)
        self.context_model: nn.Module = AutoModel.from_pretrained(cfg.model_name_or_path, config = self.config)
        self.classifier: nn.Linear = nn.Linear(self.config.hidden_size*2, 2)

    def forward(
            self,
            query_ids: T,
            query_segments: T,
            query_attention_mask: T,
            context_ids: T,
            context_segments: T,
            context_attention_mask: T,
            encoder_type: str = None,
            representation_token_pos: int = 0,
    ) -> Tuple[T, T]:
        query_output = self.query_model(
            input_ids=query_ids,
            token_type_ids=query_segments,
            attention_mask=query_attention_mask,
        )
        context_output = self.context_model(
            input_ids=context_ids,
            token_type_ids=context_segments,
            attention_mask=context_attention_mask,
        )
        query_pooled = query_output.last_hidden_state[:, 0, :]
        context_pooled = context_output.last_hidden_state[:, 0, :]
        output = self.classifier(torch.cat([query_pooled, context_pooled], dim=1))

        return output

    def create_biencoder_input(
            self,
            batch: Dict,
            tensorizer: Tensorizer,
    ) -> BiEncoderBatch:

        query_passages: Dict = batch["query"]
        context_passages: Dict = batch["context"]
        labels: T = batch["label"]

        query_tensors = []
        context_tensors = []

        for i in range(len(labels)):
            query_tensors.append(tensorizer.text_to_tensor(query_passages["abstract"][i]))
            context_tensors.append(tensorizer.text_to_tensor(context_passages["abstract"][i]))

        # passage_tensors.size() = [batch_size, max_seq_len]
        context_tensors = torch.cat([ctx.view(1, -1) for ctx in context_tensors], dim=0)
        # query_tensors.size() = [batch_size, max_seq_len]
        query_tensors = torch.cat([q.view(1, -1) for q in query_tensors], dim=0)

        context_segments = torch.zeros_like(context_tensors)
        query_segments = torch.zeros_like(query_tensors)

        return BiEncoderBatch(
            query_ids=query_tensors,
            query_segments=query_segments,
            context_ids=context_tensors,
            context_segments=context_segments,
            labels=labels,
            encoder_type="query"
        )

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        self.load_state_dict(saved_state.model_dict, strict=strict)

    def get_state_dict(self):
        return self.state_dict()



class AutoBiEncoderSum(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model_name_or_path)
        self.query_model: nn.Module = AutoModel.from_pretrained(cfg.model_name_or_path, config = self.config)
        self.context_model: nn.Module = AutoModel.from_pretrained(cfg.model_name_or_path, config = self.config)
        self.classifier: nn.Linear = nn.Linear(self.config.hidden_size, 2)

    def forward(
            self,
            query_ids: T,
            query_segments: T,
            query_attention_mask: T,
            context_ids: T,
            context_segments: T,
            context_attention_mask: T,
            encoder_type: str = None,
            representation_token_pos: int = 0,
    ) -> Tuple[T, T]:
        query_output = self.query_model(
            input_ids=query_ids,
            token_type_ids=query_segments,
            attention_mask=query_attention_mask,
        )
        context_output = self.context_model(
            input_ids=context_ids,
            token_type_ids=context_segments,
            attention_mask=context_attention_mask,
        )
        query_pooled = query_output.last_hidden_state[:, 0, :]
        context_pooled = context_output.last_hidden_state[:, 0, :]
        output = self.classifier(query_pooled + context_pooled)

        return output

    def create_biencoder_input(
            self,
            batch: Dict,
            tensorizer: Tensorizer,
    ) -> BiEncoderBatch:

        query_passages: Dict = batch["query"]
        context_passages: Dict = batch["context"]
        labels: T = batch["label"]

        query_tensors = []
        context_tensors = []

        for i in range(len(labels)):
            query_tensors.append(tensorizer.text_to_tensor(query_passages["abstract"][i]))
            context_tensors.append(tensorizer.text_to_tensor(context_passages["abstract"][i]))

        # passage_tensors.size() = [batch_size, max_seq_len]
        context_tensors = torch.cat([ctx.view(1, -1) for ctx in context_tensors], dim=0)
        # query_tensors.size() = [batch_size, max_seq_len]
        query_tensors = torch.cat([q.view(1, -1) for q in query_tensors], dim=0)

        context_segments = torch.zeros_like(context_tensors)
        query_segments = torch.zeros_like(query_tensors)

        return BiEncoderBatch(
            query_ids=query_tensors,
            query_segments=query_segments,
            context_ids=context_tensors,
            context_segments=context_segments,
            labels=labels,
            encoder_type="query"
        )

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        self.load_state_dict(saved_state.model_dict, strict=strict)

    def get_state_dict(self):
        return self.state_dict()


class AutoBiEncoderProduct(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model_name_or_path)
        self.query_model: nn.Module = AutoModel.from_pretrained(cfg.model_name_or_path, config = self.config)
        self.context_model: nn.Module = AutoModel.from_pretrained(cfg.model_name_or_path, config = self.config)
        self.classifier: nn.Linear = nn.Linear(self.config.hidden_size, 2)

    def forward(
            self,
            query_ids: T,
            query_segments: T,
            query_attention_mask: T,
            context_ids: T,
            context_segments: T,
            context_attention_mask: T,
            encoder_type: str = None,
            representation_token_pos: int = 0,
    ) -> Tuple[T, T]:
        query_output = self.query_model(
            input_ids=query_ids,
            token_type_ids=query_segments,
            attention_mask=query_attention_mask,
        )
        context_output = self.context_model(
            input_ids=context_ids,
            token_type_ids=context_segments,
            attention_mask=context_attention_mask,
        )
        query_pooled = query_output.last_hidden_state[:, 0, :]
        context_pooled = context_output.last_hidden_state[:, 0, :]
        output = self.classifier(query_pooled * context_pooled)

        return output

    def create_biencoder_input(
            self,
            batch: Dict,
            tensorizer: Tensorizer,
    ) -> BiEncoderBatch:

        query_passages: Dict = batch["query"]
        context_passages: Dict = batch["context"]
        labels: T = batch["label"]

        query_tensors = []
        context_tensors = []

        for i in range(len(labels)):
            query_tensors.append(tensorizer.text_to_tensor(query_passages["abstract"][i]))
            context_tensors.append(tensorizer.text_to_tensor(context_passages["abstract"][i]))

        # passage_tensors.size() = [batch_size, max_seq_len]
        context_tensors = torch.cat([ctx.view(1, -1) for ctx in context_tensors], dim=0)
        # query_tensors.size() = [batch_size, max_seq_len]
        query_tensors = torch.cat([q.view(1, -1) for q in query_tensors], dim=0)

        context_segments = torch.zeros_like(context_tensors)
        query_segments = torch.zeros_like(query_tensors)

        return BiEncoderBatch(
            query_ids=query_tensors,
            query_segments=query_segments,
            context_ids=context_tensors,
            context_segments=context_segments,
            labels=labels,
            encoder_type="query"
        )

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        self.load_state_dict(saved_state.model_dict, strict=strict)

    def get_state_dict(self):
        return self.state_dict()
















class BiEncoder(nn.Module):
    """
    Bi-Encoder model component
    Adopts 2 BERT models as two seperate encoders for query and passage encoding.
    """

    def __init__(
            self,
            query_model: BertEncoder,
            passage_model: BertEncoder,
            fix_q_encoder: bool = False,
            fix_p_encoder: bool = False,
    ):
        super(BiEncoder, self).__init__()
        self.query_model = query_model
        self.passage_model = passage_model
        self.classifier = nn.Linear(query_model.output_size*2, 2)
        self.fix_q_encoder = fix_q_encoder
        self.fix_p_encoder = fix_p_encoder

    @staticmethod
    def get_representation(
            sub_model: BertEncoder,
            input_ids: T,
            input_segments: T,
            attention_mask: T,
            fix_encoder: bool = False,
            representation_token_pos: int = 0,
    ) -> Tuple[T, T, T]:

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
    ) -> T:

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
        output = self.classifier(torch.cat([q_pooled_out, p_pooled_out], dim=1))

        return output

    def create_biencoder_input(
            self,
            batch: Dict,
            tensorizer: Tensorizer,
    ) -> BiEncoderBatch:

        query_passages: Dict = batch["query"]
        context_passages: Dict = batch["context"]
        labels: T = batch["label"]

        query_tensors = []
        context_tensors = []

        for i in range(len(labels)):
            query_tensors.append(tensorizer.text_to_tensor(query_passages["abstract"][i]))
            context_tensors.append(tensorizer.text_to_tensor(context_passages["abstract"][i]))

        # passage_tensors.size() = [batch_size, max_seq_len]
        context_tensors = torch.cat([ctx.view(1, -1) for ctx in context_tensors], dim=0)
        # query_tensors.size() = [batch_size, max_seq_len]
        query_tensors = torch.cat([q.view(1, -1) for q in query_tensors], dim=0)

        context_segments = torch.zeros_like(context_tensors)
        query_segments = torch.zeros_like(query_tensors)

        return BiEncoderBatch(
            query_ids=query_tensors,
            query_segments=query_segments,
            context_ids=context_tensors,
            context_segments=context_segments,
            labels=labels,
            encoder_type="query"
        )

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        self.load_state_dict(saved_state.model_dict, strict=strict)

    def get_state_dict(self):
        return self.state_dict()


class BiEncoderLoss(object):

    @staticmethod
    def get_scores(q_vectors: T, p_vectors: T) -> T:
        return dot_product_scores(q_vectors, p_vectors)
        # return size = (num_q, num_p)

    def calculate(
            self,
            scores: T,
            labels: T,
            loss_scale: float = None,
    ) -> Tuple[T, int]:
        # scores.size() = [batch_size, 2]
        softmax_scores = nn.functional.softmax(scores, dim=1)

        # labels.size() = [batch_size]
        loss = nn.functional.cross_entropy(softmax_scores, labels, reduction='mean')

        predictions: T = torch.argmax(softmax_scores, dim=1)
        correct_predictions_count = (predictions == labels).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count
