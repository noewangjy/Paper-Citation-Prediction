import logging
import torch
import transformers
from torch import Tensor as T
import torch.nn as nn
from typing import Tuple, List

from transformers import BertConfig, BertModel, AutoConfig, AutoModel
from transformers import AdamW
from transformers import BertTokenizer, AutoTokenizer

from .data_utils import Tensorizer

logger = logging.getLogger(__name__)


class BertEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = BertConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        self.bert = BertModel.from_pretrained(
            args.model_name_or_path,
            config=self.config,
            cache_dir=args.cache_dir if args.cache_dir else None
        )
        self.linear = nn.Linear(args.hidden_size, args.project_dim) if args.project_dim != 0 else None
        self.bert.init_weights()

    @property
    def output_size(self):
        if self.linear:
            return self.args.project_dim
        return self.args.hidden_size

    def forward(
            self,
            input_ids: T,
            token_type_ids: T,
            attention_mask: T,
            representation_token_pos: int = 0,
    ) -> Tuple[T, ...]:

        output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        sequence_output = output.last_hidden_state
        hidden_states = output.hidden_states
        pooled_output = output.pooler_output
        # sequence_output.size() = (batch_size, max_seq_len, hidden_size)
        # pooled_output = pooled_output[:, representation_token_pos, :]

        if self.linear:
            pooled_output = self.linear(pooled_output)

        return sequence_output, pooled_output, hidden_states


class BertTensorizer(Tensorizer):
    def __init__(self,
                 args,
                 max_seq_len: int,
                 pad_to_max: bool = True,
                 ) -> None:
        self.args = args
        self.config = BertConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            args.model_name_or_path,
            config=self.config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        self.max_seq_len = max_seq_len
        self.pad_to_max = pad_to_max

    def add_special_tokens(self, special_tokens: List[str]):
        special_tokens_num = len(special_tokens)
        logger.info("Adding %d special tokens: %s", special_tokens_num, special_tokens)
        logger.info("Tokenizer: %s", type(self.tokenizer))
        assert special_tokens_num < 500
        unused_ids = [self.tokenizer.vocab["[unused{}]".format(i)] for i in range(special_tokens_num)]
        logger.info("Utilizing the following unused token ids %s", unused_ids)
        for idx, id in enumerate(unused_ids):
            old_token = "[unused{}]".format(idx)
            del self.tokenizer.vocab[old_token]
            new_token = special_tokens[idx]
            self.tokenizer.vocab[new_token] = id
            self.tokenizer.ids_to_tokens[id] = new_token
            logging.debug("new token %s id=%s", new_token, id)

        self.tokenizer.additional_special_tokens = list(special_tokens)
        logger.info("Additional_special_tokens %s", self.tokenizer.additional_special_tokens)
        logger.info("All_special_tokens_extended: %s", self.tokenizer.all_special_tokens_extended)
        logger.info("Additional_special_tokens_ids: %s", self.tokenizer.additional_special_tokens_ids)
        logger.info("All_special_tokens %s", self.tokenizer.all_special_tokens)

    def text_to_tensor(self,
                       text: str,
                       add_special_tokens: bool = True,
                       apply_max_len: bool = True
                       ):
        text = text.strip()
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=self.max_seq_len,
            padding=False,
            truncation=True
        )

        if self.pad_to_max and len(token_ids) < self.max_seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(token_ids))

        if len(token_ids) >= self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len] if apply_max_len else token_ids
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attention_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

    def get_token_id(self, token: str) -> int:
        return self.tokenizer.vocab[token]


# TODO: Modify according to BertEncoder if it works

class AutoEncoder(AutoModel):
    def __init__(self, config, project_dim: int = 0):
        AutoModel().__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero!"
        self.linear = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.ini_weights()

    @classmethod
    def init_encoder(
            cls,
            config_name: str,
            project_dim: int = 0,
            dropout: float = 0.1,
            pretrained: bool = True,
            **kwargs
    ) -> AutoModel:
        logger.info("Initializing BERT Encoder from config: %s", config_name)
        config = AutoConfig.from_pretrained(config_name if config_name else "bert-base-uncased")

        if dropout != 0:
            config.attention_probs_dropout_prob = dropout
            config.hidden_dropout_prob = dropout

        if pretrained:
            return cls.from_pretrained(config_name, config=config, project_dim=project_dim, **kwargs)
        else:
            return AutoEncoder(config=config, project_dim=project_dim)

    def forward(
            self,
            input_ids: T,
            token_type_ids: T,
            attention_mask: T,
            representation_token_pos: int = 0,
    ) -> Tuple[T, ...]:

        output = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        sequence_output = output.last_hidden_state
        hidden_states = output.hidden_states
        # sequence_output.size() = (batch_size, max_seq_len, hidden_size)
        pooled_output = sequence_output[:, representation_token_pos, :]

        if self.linear:
            pooled_output = self.linear(pooled_output)

        return sequence_output, pooled_output, hidden_states

    def get_output_size(self):
        if self.linear:
            return self.linear.out_features
        return self.config.hidden_size


class AutoTensorizer(Tensorizer):
    def __init__(self,
                 tokenizer: AutoTokenizer,
                 max_seq_len: int,
                 pad_to_max: bool = True,
                 ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_to_max = pad_to_max

    def text_to_tensor(self,
                       text: str,
                       add_special_tokens: bool = True,
                       apply_max_len: bool = True
                       ):
        text = text.strip()
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=self.max_seq_len,
            padding=False,
            truncation=True
        )

        if self.pad_to_max and len(token_ids) < self.max_seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(token_ids))

        if len(token_ids) >= self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len] if apply_max_len else token_ids
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attention_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

    def get_token_id(self, token: str) -> int:
        return self.tokenizer.vocab[token]


def get_optimizer_grouped(
        optimizer_grouped_parameters: List,
        learning_rate: float = 1e-5,
        adam_eps: float = 1e-8,
) -> torch.optim.Optimizer:
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer


def get_hf_model_param_grouping(
        model: nn.Module,
        weight_decay: float = 0.0,
):
    no_decay = ["bias", "LayerNorm.weight"]

    return [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


