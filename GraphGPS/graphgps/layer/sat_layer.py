# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import nn
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
from einops import rearrange
import torch.nn.functional as F

# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric import utils
from torch_scatter import scatter
from torch.nn import ModuleList, Sequential, Linear, ReLU
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from typing import Optional, List, Dict
from torch_geometric.nn.inits import reset
from torch_geometric.utils import degree

# -*- coding: utf-8 -*-
from typing import Optional
import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr
from torch_geometric.utils.num_nodes import maybe_num_nodes
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np


def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def dense_to_sparse_tensor(matrix):
    rows, columns = torch.where(matrix > 0)
    values = torch.ones(rows.shape)
    indices = torch.from_numpy(np.vstack((rows,
                                          columns))).long()
    shape = torch.Size(matrix.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def extract_node_feature(data, reduce='add'):
    if reduce in ['mean', 'max', 'add']:
        data.x = scatter(data.edge_attr,
                         data.edge_index[0],
                         dim=0,
                         dim_size=data.num_nodes,
                         reduce=reduce)
    else:
        raise Exception('Unknown Aggregation Type')
    return data


def pad_batch(x, ptr, return_mask=False):
    bsz = len(ptr) - 1
    # num_nodes = torch.diff(ptr)
    max_num_nodes = torch.diff(ptr).max().item()

    all_num_nodes = ptr[-1].item()
    cls_tokens = False
    x_size = len(x[0]) if isinstance(x, (list, tuple)) else len(x)
    if x_size > all_num_nodes:
        cls_tokens = True
        max_num_nodes += 1
    if isinstance(x, (list, tuple)):
        new_x = [xi.new_zeros(bsz, max_num_nodes, xi.shape[-1]) for xi in x]
        if return_mask:
            padding_mask = x[0].new_zeros(bsz, max_num_nodes).bool()
    else:
        new_x = x.new_zeros(bsz, max_num_nodes, x.shape[-1])
        if return_mask:
            padding_mask = x.new_zeros(bsz, max_num_nodes).bool()

    for i in range(bsz):
        num_node = ptr[i + 1] - ptr[i]
        if isinstance(x, (list, tuple)):
            for j in range(len(x)):
                new_x[j][i][:num_node] = x[j][ptr[i]:ptr[i + 1]]
                if cls_tokens:
                    new_x[j][i][-1] = x[j][all_num_nodes + i]
        else:
            new_x[i][:num_node] = x[ptr[i]:ptr[i + 1]]
            if cls_tokens:
                new_x[i][-1] = x[all_num_nodes + i]
        if return_mask:
            padding_mask[i][num_node:] = True
            if cls_tokens:
                padding_mask[i][-1] = False
    if return_mask:
        return new_x, padding_mask
    return new_x


def unpad_batch(x, ptr):
    bsz, n, d = x.shape
    max_num_nodes = torch.diff(ptr).max().item()
    num_nodes = ptr[-1].item()
    all_num_nodes = num_nodes
    cls_tokens = False
    if n > max_num_nodes:
        cls_tokens = True
        all_num_nodes += bsz
    new_x = x.new_zeros(all_num_nodes, d)
    for i in range(bsz):
        new_x[ptr[i]:ptr[i + 1]] = x[i][:ptr[i + 1] - ptr[i]]
        if cls_tokens:
            new_x[num_nodes + i] = x[i][-1]
    return new_x


GNN_TYPES = [
    'graph', 'graphsage', 'gcn',
    'gin', 'gine',
    'pna', 'pna2', 'pna3', 'mpnn', 'pna4',
    'rwgnn', 'khopgnn'
]

EDGE_GNN_TYPES = [
    'gine', 'gcn',
    'pna', 'pna2', 'pna3', 'mpnn', 'pna4'
]


def get_simple_gnn_layer(gnn_type, embed_dim, **kwargs):
    edge_dim = kwargs.get('edge_dim', None)
    if gnn_type == "graph":
        return gnn.GraphConv(embed_dim, embed_dim)
    elif gnn_type == "graphsage":
        return gnn.SAGEConv(embed_dim, embed_dim)
    elif gnn_type == "gcn":
        if edge_dim is None:
            return gnn.GCNConv(embed_dim, embed_dim)
        else:
            return GCNConv(embed_dim, edge_dim)
    elif gnn_type == "gin":
        mlp = mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )
        return gnn.GINConv(mlp, train_eps=True)
    elif gnn_type == "gine":
        mlp = mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )
        return gnn.GINEConv(mlp, train_eps=True, edge_dim=edge_dim)
    elif gnn_type == "pna":
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        deg = kwargs.get('deg', None)
        layer = gnn.PNAConv(embed_dim, embed_dim,
                            aggregators=aggregators, scalers=scalers,
                            deg=deg, towers=4, pre_layers=1, post_layers=1,
                            divide_input=True, edge_dim=edge_dim)
        return layer
    elif gnn_type == "pna2":
        aggregators = ['mean', 'sum', 'max']
        scalers = ['identity']
        deg = kwargs.get('deg', None)
        layer = gnn.PNAConv(embed_dim, embed_dim,
                            aggregators=aggregators, scalers=scalers,
                            deg=deg, towers=4, pre_layers=1, post_layers=1,
                            divide_input=True, edge_dim=edge_dim)
        return layer
    # elif gnn_type == "pna2_ram":
    #     aggregators = ['mean', 'sum', 'max']
    #     scalers = ['identity']
    #     deg = kwargs.get('deg', None)
    #     layer = PNAConv_towers(embed_dim, embed_dim,
    #                 aggregators=aggregators, scalers=scalers,
    #                 deg=deg, towers=4, divide_input=True, edge_dim=edge_dim)
    #     return layer
    elif gnn_type == "pna3":
        aggregators = ['mean', 'sum', 'max']
        scalers = ['identity']
        deg = kwargs.get('deg', None)
        layer = gnn.PNAConv(embed_dim, embed_dim,
                            aggregators=aggregators, scalers=scalers,
                            deg=deg, towers=1, pre_layers=1, post_layers=1,
                            divide_input=False, edge_dim=edge_dim)
        return layer
    elif gnn_type == "pna4":
        aggregators = ['mean', 'sum', 'max']
        scalers = ['identity']
        deg = kwargs.get('deg', None)
        layer = gnn.PNAConv(embed_dim, embed_dim,
                            aggregators=aggregators, scalers=scalers,
                            deg=deg, towers=8, pre_layers=1, post_layers=1,
                            divide_input=True, edge_dim=edge_dim)
        return layer


    elif gnn_type == "mpnn":
        aggregators = ['sum']
        scalers = ['identity']
        deg = kwargs.get('deg', None)
        layer = gnn.PNAConv(embed_dim, embed_dim,
                            aggregators=aggregators, scalers=scalers,
                            deg=deg, towers=4, pre_layers=1, post_layers=1,
                            divide_input=True, edge_dim=edge_dim)
        return layer
    else:
        raise ValueError("Not implemented!")


class GCNConv(gnn.MessagePassing):
    def __init__(self, embed_dim, edge_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = nn.Linear(embed_dim, embed_dim)
        self.root_emb = nn.Embedding(1, embed_dim)

        # edge_attr is two dimensional after augment_edge transformation
        self.edge_encoder = nn.Linear(edge_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = utils.degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(
            edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(
            x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class Attention(gnn.MessagePassing):
    """Multi-head Structure-Aware attention using PyG interface
    accept Batch data given by PyG

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    num_heads (int):        number of attention heads (default: 8)
    dropout (float):        dropout value (default: 0.0)
    bias (bool):            whether layers have an additive bias (default: False)
    symmetric (bool):       whether K=Q in dot-product attention (default: False)
    gnn_type (str):         GNN type to use in structure extractor. (see gnn_layers.py for options)
    se (str):               type of structure extractor ("gnn", "khopgnn")
    k_hop (int):            number of base GNN layers or the K hop size for khopgnn structure extractor (default=2).
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0., bias=False,
                 symmetric=False, gnn_type="gcn", se="gnn", k_hop=2, **kwargs):

        super().__init__(node_dim=0, aggr='add')
        self.embed_dim = embed_dim
        self.bias = bias
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.se = se

        self.gnn_type = gnn_type
        if self.se == "khopgnn":
            self.khop_structure_extractor = KHopStructureExtractor(embed_dim, gnn_type=gnn_type,
                                                                   num_layers=k_hop, **kwargs)
        else:
            self.structure_extractor = StructureExtractor(embed_dim, gnn_type=gnn_type,
                                                          num_layers=k_hop, **kwargs)
        self.attend = nn.Softmax(dim=-1)

        self.symmetric = symmetric
        if symmetric:
            self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

        self.attn_sum = None

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_qk.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        if self.bias:
            nn.init.constant_(self.to_qk.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self,
                x,
                edge_index,
                complete_edge_index,
                subgraph_node_index=None,
                subgraph_edge_index=None,
                subgraph_indicator_index=None,
                subgraph_edge_attr=None,
                edge_attr=None,
                ptr=None,
                return_attn=False):
        """
        Compute attention layer.

        Args:
        ----------
        x:                          input node features
        edge_index:                 edge index from the graph
        complete_edge_index:        edge index from fully connected graph
        subgraph_node_index:        documents the node index in the k-hop subgraphs
        subgraph_edge_index:        edge index of the extracted subgraphs
        subgraph_indicator_index:   indices to indicate to which subgraph corresponds to which node
        subgraph_edge_attr:         edge attributes of the extracted k-hop subgraphs
        edge_attr:                  edge attributes
        return_attn:                return attention (default: False)

        """
        # Compute value matrix

        v = self.to_v(x)

        # Compute structure-aware node embeddings
        if self.se == 'khopgnn':  # k-subgraph SAT
            x_struct = self.khop_structure_extractor(
                x=x,
                edge_index=edge_index,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_indicator_index=subgraph_indicator_index,
                subgraph_node_index=subgraph_node_index,
                subgraph_edge_attr=subgraph_edge_attr,
            )
        else:  # k-subtree SAT
            x_struct = self.structure_extractor(x, edge_index, edge_attr)

        # Compute query and key matrices
        if self.symmetric:
            qk = self.to_qk(x_struct)
            qk = (qk, qk)
        else:
            qk = self.to_qk(x_struct).chunk(2, dim=-1)

        # Compute complete self-attention
        attn = None

        if complete_edge_index is not None:
            out = self.propagate(complete_edge_index, v=v, qk=qk, edge_attr=None, size=None,
                                 return_attn=return_attn)
            if return_attn:
                attn = self._attn
                self._attn = None
                attn = torch.sparse_coo_tensor(
                    complete_edge_index,
                    attn,
                ).to_dense().transpose(0, 1)

            out = rearrange(out, 'n h d -> n (h d)')
        else:
            out, attn = self.self_attn(qk, v, ptr, return_attn=return_attn)
        return self.out_proj(out), attn

    def message(self, v_j, qk_j, qk_i, edge_attr, index, ptr, size_i, return_attn):
        """Self-attention operation compute the dot-product attention """

        qk_i = rearrange(qk_i, 'n (h d) -> n h d', h=self.num_heads)
        qk_j = rearrange(qk_j, 'n (h d) -> n h d', h=self.num_heads)
        v_j = rearrange(v_j, 'n (h d) -> n h d', h=self.num_heads)
        attn = (qk_i * qk_j).sum(-1) * self.scale
        if edge_attr is not None:
            attn = attn + edge_attr
        attn = utils.softmax(attn, index, ptr, size_i)
        if return_attn:
            self._attn = attn
        attn = self.attn_dropout(attn)

        return v_j * attn.unsqueeze(-1)

    def self_attn(self, qk, v, ptr, return_attn=False):
        """ Self attention which can return the attn """

        qk, mask = pad_batch(qk, ptr, return_mask=True)
        k, q = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qk)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots = dots.masked_fill(
            mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )

        dots = self.attend(dots)
        dots = self.attn_dropout(dots)

        v = pad_batch(v, ptr)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = unpad_batch(out, ptr)

        if return_attn:
            return out, dots
        return out, None


class StructureExtractor(nn.Module):
    r""" K-subtree structure extractor. Computes the structure-aware node embeddings using the
    k-hop subtree centered around each node.

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    gnn_type (str):         GNN type to use in structure extractor. (gcn, gin, pna, etc)
    num_layers (int):       number of GNN layers
    batch_norm (bool):      apply batch normalization or not
    concat (bool):          whether to concatenate the initial edge features
    khopgnn (bool):         whether to use the subgraph instead of subtree
    """

    def __init__(self, embed_dim, gnn_type="gcn", num_layers=3,
                 batch_norm=True, concat=True, khopgnn=False, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.khopgnn = khopgnn
        self.concat = concat
        self.gnn_type = gnn_type
        layers = []
        for _ in range(num_layers):
            layers.append(get_simple_gnn_layer(gnn_type, embed_dim, **kwargs))
        self.gcn = nn.ModuleList(layers)

        self.relu = nn.ReLU()
        self.batch_norm = batch_norm
        inner_dim = (num_layers + 1) * embed_dim if concat else embed_dim

        if batch_norm:
            self.bn = nn.BatchNorm1d(inner_dim)

        self.out_proj = nn.Linear(inner_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr=None,
                subgraph_indicator_index=None, agg="sum"):
        x_cat = [x]
        for gcn_layer in self.gcn:
            # if self.gnn_type == "attn":
            #     x = gcn_layer(x, edge_index, None, edge_attr=edge_attr)
            if self.gnn_type in EDGE_GNN_TYPES:
                if edge_attr is None:
                    x = self.relu(gcn_layer(x, edge_index))
                else:
                    x = self.relu(gcn_layer(x, edge_index, edge_attr=edge_attr))
            else:
                x = self.relu(gcn_layer(x, edge_index))

            if self.concat:
                x_cat.append(x)

        if self.concat:
            x = torch.cat(x_cat, dim=-1)

        if self.khopgnn:
            if agg == "sum":
                x = scatter_add(x, subgraph_indicator_index, dim=0)
            elif agg == "mean":
                x = scatter_mean(x, subgraph_indicator_index, dim=0)
            return x

        if self.num_layers > 0 and self.batch_norm:
            x = self.bn(x)

        x = self.out_proj(x)
        return x


class KHopStructureExtractor(nn.Module):
    r""" K-subgraph structure extractor. Extracts a k-hop subgraph centered around
    each node and uses a GNN on each subgraph to compute updated structure-aware
    embeddings.

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    gnn_type (str):         GNN type to use in structure extractor. (gcn, gin, pna, etc)
    num_layers (int):       number of GNN layers
    concat (bool):          whether to concatenate the initial edge features
    khopgnn (bool):         whether to use the subgraph instead of subtree (True)
    """

    def __init__(self, embed_dim, gnn_type="gcn", num_layers=3, batch_norm=True,
                 concat=True, khopgnn=True, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.khopgnn = khopgnn

        self.batch_norm = batch_norm

        self.structure_extractor = StructureExtractor(
            embed_dim,
            gnn_type=gnn_type,
            num_layers=num_layers,
            concat=False,
            khopgnn=True,
            **kwargs
        )

        if batch_norm:
            self.bn = nn.BatchNorm1d(2 * embed_dim)

        self.out_proj = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, x, edge_index, subgraph_edge_index, edge_attr=None,
                subgraph_indicator_index=None, subgraph_node_index=None,
                subgraph_edge_attr=None):

        x_struct = self.structure_extractor(
            x=x[subgraph_node_index],
            edge_index=subgraph_edge_index,
            edge_attr=subgraph_edge_attr,
            subgraph_indicator_index=subgraph_indicator_index,
            agg="sum",
        )
        x_struct = torch.cat([x, x_struct], dim=-1)
        if self.batch_norm:
            x_struct = self.bn(x_struct)
        x_struct = self.out_proj(x_struct)

        return x_struct


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    r"""Structure-Aware Transformer layer, made up of structure-aware self-attention and feed-forward network.

    Args:
    ----------
        d_model (int):      the number of expected features in the input (required).
        nhead (int):        the number of heads in the multiheadattention models (default=8).
        dim_feedforward (int): the dimension of the feedforward network model (default=512).
        dropout:            the dropout value (default=0.1).
        activation:         the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable (default: relu).
        batch_norm:         use batch normalization instead of layer normalization (default: True).
        pre_norm:           pre-normalization or post-normalization (default=False).
        gnn_type:           base GNN model to extract subgraph representations.
                            One can implememnt customized GNN in gnn_layers.py (default: gcn).
        se:                 structure extractor to use, either gnn or khopgnn (default: gnn).
        k_hop:              the number of base GNN layers or the K hop size for khopgnn structure extractor (default=2).
    """

    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1,
                 activation="relu", batch_norm=True, pre_norm=False,
                 gnn_type="gcn", se="gnn", k_hop=2, **kwargs):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

        self.self_attn = Attention(d_model, nhead, dropout=dropout,
                                   bias=False, gnn_type=gnn_type, se=se, k_hop=k_hop, **kwargs)
        self.batch_norm = batch_norm
        self.pre_norm = pre_norm
        if batch_norm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)

    def forward(self, x, edge_index, complete_edge_index,
                subgraph_node_index=None, subgraph_edge_index=None,
                subgraph_edge_attr=None,
                subgraph_indicator_index=None,
                edge_attr=None, degree=None, ptr=None,
                return_attn=False,
                ):

        if self.pre_norm:
            x = self.norm1(x)

        x2, attn = self.self_attn(
            x,
            edge_index,
            complete_edge_index,
            edge_attr=edge_attr,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_edge_attr=subgraph_edge_attr,
            ptr=ptr,
            return_attn=return_attn
        )

        if degree is not None:
            x2 = degree.unsqueeze(-1) * x2
        x = x + self.dropout1(x2)
        if self.pre_norm:
            x = self.norm2(x)
        else:
            x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)

        if not self.pre_norm:
            x = self.norm2(x)
        return x
