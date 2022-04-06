import torch
from torch import nn as nn
from transformers import AutoModel


class MLPBert(nn.Module):
    def __init__(self,
                 bert_model_name: str = 'bert-base-uncased',
                 bert_num_feature: int = 768,
                 hidden_dim: int = 128):
        super().__init__()
        # self.author_model = AutoModel.from_pretrained(bert_model_name)
        self.abstract_model = AutoModel.from_pretrained(bert_model_name)
        self.mlp_input_dim = 3 + 2 * bert_num_feature  # TODO: Find a way to generate author embeddings
        self.bn1 = nn.BatchNorm1d(self.mlp_input_dim)
        self.l1 = nn.Linear(self.mlp_input_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 2)

    def forward(self, u_deg, v_deg, uv_deg_diff, u_authors, v_authors, u_abstracts, v_abstracts):
        # batch_size = u_deg.shape[0]
        # u_authors_embed = self.author_model(u_authors, torch.zeros_like(u_authors)).last_hidden_state[:, 0, :]
        # v_authors_embed = self.author_model(v_authors, torch.zeros_like(v_authors)).last_hidden_state[:, 0, :] # TODO: Find a way to generate author embeddings
        u_abstracts_embed = self.abstract_model(u_abstracts, torch.zeros_like(u_abstracts)).last_hidden_state[:, 0, :]
        v_abstracts_embed = self.abstract_model(v_abstracts, torch.zeros_like(v_abstracts)).last_hidden_state[:, 0, :]
        out = torch.cat([u_deg, v_deg, uv_deg_diff,
                         # u_authors_embed,
                         # v_authors_embed,
                         u_abstracts_embed,
                         v_abstracts_embed], dim=1)
        out = torch.relu(self.l1(self.bn1(out)))
        out = torch.relu(self.l2(self.bn2(out)))
        return out
