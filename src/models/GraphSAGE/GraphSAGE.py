import torch.nn as nn
import torch.nn.functional as F
import torch

from dgl.nn import SAGEConv
import dgl.function as fn
from transformers import AutoModel


class GraphSAGE(nn.Module):
    def __init__(self, input_dims, hidden_dims):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dims, hidden_dims, 'mean')
        self.conv2 = SAGEConv(hidden_dims, hidden_dims, 'mean')

    def forward(self, g, g_features):
        h = self.conv1(g, g_features)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class GraphSAGEBert(nn.Module):
    def __init__(self, input_dims, hidden_dims, predictor: str):
        super().__init__()
        self.author_model = AutoModel.from_pretrained("bert-base-uncased")
        self.abstract_model = AutoModel.from_pretrained("bert-base-uncased")

        self.graph_model = GraphSAGE(input_dims, hidden_dims)
        if predictor == 'dot':
            self.predictor = DotPredictor()
        elif predictor == 'mlp':
            self.predictor = MLPPredictor(hidden_dims)
        else:
            raise ValueError('Not valid predictor')

    def forward(self, x):
        raise NotImplementedError


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # 通过源节点特征“h”和目标节点特征“h”之间的点积计算两点之间存在边的Score
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v为每条边返回一个元素向量，因此需要squeeze操作
            return g.edata['score'][:, 0]


class MLPPredictor(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        self.W1 = nn.Linear(hidden_dims * 2, hidden_dims)
        self.W2 = nn.Linear(hidden_dims, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.
        """
        out = self.W1(edges)
        out = torch.relu(out)
        out = self.W2(out)
        out = torch.relu(out)
        return out
