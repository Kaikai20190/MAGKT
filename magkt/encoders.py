"""Graph encoders.

StructEncoderSAGE (default, Sec. 3.3):
    P Z    = D_G^-1 G Z                      (mean aggregation over the hypergraph operator)
    Z_hat  = W2 . Dropout(ReLU(W1 [Z || P Z]))
  The node's own embedding is concatenated with the aggregated neighbourhood (GraphSAGE mean aggregator),
  so nodes with identical hyperedge memberships (e.g. questions with the same concepts) still receive
  different representations.

StructEncoderLegacy (aggregation-only encoder, ablation 'legacy_struct'):
    Z_hat  = W2 . Dropout(ReLU(W1 (D_G^-1 G Z)))
  Nodes with identical rows of G receive identical representations.

GCN: two graph-convolution layers with ReLU, used on the directed transition graph (Sec. 3.4.1).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply(adj, x):
    """adj is a sparse matrix G or a factor pair (L, R) with G = L @ R."""
    if isinstance(adj, (tuple, list)):
        L, R = adj
        return torch.sparse.mm(L, torch.sparse.mm(R, x))
    return torch.sparse.mm(adj, x)


def _mean_aggregate(adj, x):
    h = _apply(adj, x)
    deg = _apply(adj, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype))
    deg = torch.where(deg == 0, torch.ones_like(deg), deg)
    return h / deg


class StructEncoderLegacy(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x, adj):
        h = _mean_aggregate(adj, x)
        h = F.relu(self.fc1(h))
        h = F.dropout(h, self.dropout, training=self.training)
        return self.fc2(h)


class StructEncoderSAGE(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x, adj):
        h = torch.cat([x, _mean_aggregate(adj, x)], dim=-1)
        h = F.relu(self.fc1(h))
        h = F.dropout(h, self.dropout, training=self.training)
        return self.fc2(h)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        stdv = 1.0 / math.sqrt(out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        out = torch.sparse.mm(adj, x @ self.weight)
        return out + self.bias if self.bias is not None else out


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        return F.relu(self.gc2(x, adj))
