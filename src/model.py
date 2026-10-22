"""
Heterogeneous link-prediction models for the Guizhi Tang knowledge graph.

Four models are provided so that the revision can compare a feature-only
baseline, a homogeneous GCN, a relational GCN and a heterogeneous GNN on
identical splits:

    MLPBaseline   - no message passing (compound and disease features only)
    HomogeneousGCN- one shared GCN over the union of all typed edges
    RGCN          - relational GCN with relation-specific weight matrices
    HetGNN        - type-specific input projections + per-relation graph
                    attention aggregation (GAT) with residual connections

All models use the same MLP decoder on the concatenation of the compound and
disease embeddings.  The code deliberately avoids any trainable component that
was not used in the original submission (the original `type_attention` module
was defined but never called and is not reproduced here).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, GCNConv, HeteroConv, RGCNConv, SAGEConv
from torch_geometric.transforms import ToUndirected


def build_hetero_data(graph, train_labels, device="cpu"):
    """Create a PyG HeteroData object.

    Only training compound-disease edges are inserted into the message-passing
    graph; validation/test label edges never enter the graph.
    """
    data = HeteroData()
    for node_type, feats in graph["features"].items():
        data[node_type].x = torch.tensor(feats, dtype=torch.float32, device=device)
    for relation, edge_list in graph["edges"].items():
        src_type, rel, dst_type = relation.split("-")
        if not edge_list:
            continue
        ei = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
        data[(src_type, rel, dst_type)].edge_index = ei
    if train_labels:
        ei = torch.tensor(train_labels, dtype=torch.long, device=device).t().contiguous()
        data[("compound", "may_treat", "disease")].edge_index = ei
        data[("disease", "rev_may_treat", "compound")].edge_index = ei.flip(0)
    data = ToUndirected()(data)
    return data


class MLPDecoder(nn.Module):
    def __init__(self, hidden, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, src, dst):
        return self.net(torch.cat([src, dst], dim=-1)).squeeze(-1)


class MLPBaseline(nn.Module):
    def __init__(self, in_dims, hidden=64, dropout=0.3, **kwargs):
        super().__init__()
        self.c = nn.Linear(in_dims["compound"], hidden)
        self.d = nn.Linear(in_dims["disease"], hidden)
        self.decoder = MLPDecoder(hidden, dropout)

    def forward(self, data, c_idx, d_idx):
        return self.decoder(F.relu(self.c(data["compound"].x[c_idx])),
                            F.relu(self.d(data["disease"].x[d_idx])))


class HomogeneousGCN(nn.Module):
    def __init__(self, in_dims, hidden=64, layers=2, dropout=0.3, **kwargs):
        super().__init__()
        self.proj = nn.ModuleDict({nt: nn.Linear(dim, hidden) for nt, dim in in_dims.items()})
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(layers)])
        self.decoder = MLPDecoder(hidden, dropout)
        self.dropout = dropout

    def _union(self, data):
        xs, offsets, offset = [], {}, 0
        for nt in data.node_types:
            offsets[nt] = offset
            xs.append(self.proj[nt](data[nt].x))
            offset += data[nt].num_nodes
        x = torch.cat(xs, 0)
        srcs, dsts = [], []
        for (s, r, d) in data.edge_types:
            ei = data[(s, r, d)].edge_index
            srcs.append(ei[0] + offsets[s])
            dsts.append(ei[1] + offsets[d])
        edge_index = torch.stack([torch.cat(srcs), torch.cat(dsts)], dim=0) if srcs else \
            torch.zeros((2, 0), dtype=torch.long)
        return x, edge_index, offsets

    def encode(self, data):
        x, edge_index, offsets = self._union(data)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x, offsets

    def forward(self, data, c_idx, d_idx):
        x, offsets = self.encode(data)
        return self.decoder(x[offsets["compound"] + c_idx], x[offsets["disease"] + d_idx])


class RGCN(nn.Module):
    def __init__(self, in_dims, hidden=64, layers=2, dropout=0.3, num_relations=14, **kwargs):
        super().__init__()
        self.proj = nn.ModuleDict({nt: nn.Linear(dim, hidden) for nt, dim in in_dims.items()})
        self.convs = nn.ModuleList([RGCNConv(hidden, hidden, num_relations) for _ in range(layers)])
        self.decoder = MLPDecoder(hidden, dropout)
        self.dropout = dropout
        self.num_relations = num_relations

    def _union(self, data):
        xs, offsets, offset = [], {}, 0
        for nt in data.node_types:
            offsets[nt] = offset
            xs.append(self.proj[nt](data[nt].x))
            offset += data[nt].num_nodes
        x = torch.cat(xs, 0)
        srcs, dsts, rels = [], [], []
        rel_map = {et: i for i, et in enumerate(data.edge_types)}
        for et in data.edge_types:
            s, r, d = et
            ei = data[et].edge_index
            srcs.append(ei[0] + offsets[s])
            dsts.append(ei[1] + offsets[d])
            rels.append(torch.full((ei.shape[1],), rel_map[et], dtype=torch.long))
        edge_index = torch.stack([torch.cat(srcs), torch.cat(dsts)], dim=0) if srcs else \
            torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.cat(rels) if rels else torch.zeros((0,), dtype=torch.long)
        return x, edge_index, edge_type, offsets

    def encode(self, data):
        x, edge_index, edge_type, offsets = self._union(data)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_type))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x, offsets

    def forward(self, data, c_idx, d_idx):
        x, offsets = self.encode(data)
        return self.decoder(x[offsets["compound"] + c_idx], x[offsets["disease"] + d_idx])


class HetGNN(nn.Module):
    """Heterogeneous GNN with per-relation graph attention aggregation."""

    def __init__(self, in_dims, hidden=64, layers=2, heads=4, dropout=0.3, **kwargs):
        super().__init__()
        self.proj = nn.ModuleDict({nt: nn.Linear(dim, hidden) for nt, dim in in_dims.items()})
        self.convs = nn.ModuleList()
        for _ in range(layers):
            conv = HeteroConv({
                et: GATConv((-1, -1), hidden, heads=heads, concat=False,
                            dropout=dropout, add_self_loops=False)
                for et in kwargs["edge_types"]
            }, aggr="mean")
            self.convs.append(conv)
        self.norms = nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in in_dims})
        self.decoder = MLPDecoder(hidden, dropout)
        self.dropout = dropout

    def encode(self, data):
        x = {nt: self.proj[nt](data[nt].x) for nt in data.node_types}
        for conv in self.convs:
            out = conv(x, data.edge_index_dict)
            for nt in x:
                if nt in out:
                    x[nt] = F.elu(self.norms[nt](out[nt] + x[nt]))
                    x[nt] = F.dropout(x[nt], p=self.dropout, training=self.training)
        return x

    def forward(self, data, c_idx, d_idx):
        x = self.encode(data)
        return self.decoder(x["compound"][c_idx], x["disease"][d_idx])


def build_model(name, in_dims, edge_types=None, **kwargs):
    if name == "mlp":
        return MLPBaseline(in_dims, **kwargs)
    if name == "gcn":
        return HomogeneousGCN(in_dims, **kwargs)
    if name == "rgcn":
        return RGCN(in_dims, **kwargs, num_relations=len(edge_types))
    if name == "hetgnn":
        return HetGNN(in_dims, edge_types=edge_types, **kwargs)
    raise ValueError(name)

