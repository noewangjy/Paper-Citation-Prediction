import torch.nn as nn
import torch

from dgl.nn import SAGEConv
import dgl.function as fn
from transformers import AutoModel


class GraphSAGE(nn.Module):
    def __init__(self, input_dims, hidden_dims):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dims, hidden_dims, 'lstm')
        self.conv2 = SAGEConv(hidden_dims, hidden_dims, 'lstm')
        self.conv3 = SAGEConv(hidden_dims, hidden_dims, 'lstm')

    def forward(self, g, g_features):
        h = self.conv1(g, g_features)
        h = torch.relu(h)
        h = self.conv2(g, h)
        h = torch.relu(h)
        h = self.conv3(g, h)
        return h


class GraphSAGEBert(nn.Module):
    def __init__(self,
                 bert_model_name: str,
                 bert_num_feature: int,
                 input_dims, hidden_dims, predictor: str):
        super().__init__()
        self.author_model = AutoModel.from_pretrained(bert_model_name)
        self.abstract_model = AutoModel.from_pretrained(bert_model_name)
        self.bert_num_feature = bert_num_feature

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
            return torch.sigmoid(g.edata['score'][:, 0])


class MLPPredictor(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()
        self.l1 = nn.Linear(hidden_dims * 2, hidden_dims)
        self.l2 = nn.Linear(hidden_dims, 1)
        self.l3 = nn.Linear(hidden_dims, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.
        """
        out = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        out = torch.relu(self.l1(out))
        out = torch.relu(self.l2(out))
        out = torch.relu(self.l3(out))
        return {'score': out}

    def forward(self, g, h):
        # h contains the node representations computed from the GNN defined in node_clf.py
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return torch.sigmoid(g.edata['score'])


class GraphSAGEBundled(nn.Module):
    def __init__(self,
                 input_dims,
                 hidden_dims,
                 predictor: str = 'dot',
                 aggregator: str = 'mean'):
        super(GraphSAGEBundled, self).__init__()
        self.hidden_dims = hidden_dims
        self.aggregator = aggregator
        self.conv1 = SAGEConv(input_dims, hidden_dims, aggregator)
        self.conv2 = SAGEConv(hidden_dims, hidden_dims, aggregator)
        self.conv3 = SAGEConv(hidden_dims, hidden_dims, aggregator)

        if predictor == 'dot':
            self.predictor = DotPredictor()
        elif predictor == 'mlp':
            self.predictor = MLPPredictor(hidden_dims)
        else:
            raise ValueError('Not valid predictor')
        self.hidden_state = None

    def forward(self, g, g_features):
        h = self.conv1(g, g_features)
        h = torch.relu(h)
        h = self.conv2(g, h)
        h = torch.relu(h)
        h = self.conv3(g, h)
        self.hidden_state = h
        return h
