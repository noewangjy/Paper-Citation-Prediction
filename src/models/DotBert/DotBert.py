import torch
from torch import nn as nn
from transformers import AutoModel


class DotBert(nn.Module):
    def __init__(self,
                 bert_model_name: str = 'bert-base-uncased',
                 bert_num_feature: int = 768,
                 loss_type: str = "mse"):
        super().__init__()
        self.abstract_model = AutoModel.from_pretrained(bert_model_name)
        self.metrics = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        self.loss_type = loss_type

    def forward(self, u_abstracts, v_abstracts):
        u_abstracts_embed = self.abstract_model(u_abstracts, torch.zeros_like(u_abstracts)).pooler_output
        v_abstracts_embed = self.abstract_model(v_abstracts, torch.zeros_like(v_abstracts)).pooler_output
        cos_sim = self.metrics(u_abstracts_embed, v_abstracts_embed)
        if self.loss_type == "bce":
            return torch.div(torch.add(cos_sim, 1), 2).view(-1, 1)
        elif self.loss_type == "mse":
            return cos_sim.view(-1, 1)
        else:
            raise NotImplementedError