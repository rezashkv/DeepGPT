from typing import Optional

from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch
from transformers import PreTrainedModel
from torch_scatter import scatter
from transformers.modeling_outputs import SequenceClassifierOutput

from .configuration_gps import GraphGPSConfig

from .layer import SingleBigBirdLayer
from .layer import GatedGCNLayer
from .layer import GINEConvESLapPE
from .prefix_encoder import GraphGPSPrefixEncoder


def global_mean_pool(x, batch, size=None):
    """
    Globally pool node embeddings into graph embeddings, via elementwise mean.
    Pooling function takes in node embedding [num_nodes x emb_dim] and
    batch (indices) and outputs graph embedding [num_graphs x emb_dim].

    Args:
        x (torch.tensor): Input node embeddings
        batch (torch.tensor): Batch tensor that indicates which node
        belongs to which graph
        size (optional): Total number of graphs. Can be auto-inferred.

    Returns: Pooled graph embeddings

    """
    size = batch.max().item() + 1 if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce='mean')


class SANGraphHead(nn.Module):
    """
    SAN prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out, L=2):
        super().__init__()
        self.pooling_fun = global_mean_pool
        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = nn.GELU()

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch, size=batch.num_graphs)
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label


class SimpleAtomEncoder(torch.nn.Module):
    """
    The atom Encoder used in OGB molecule dataset.

    Args:
        emb_dim (int): Output embedding dimension
    """

    def __init__(self, emb_dim):
        super().__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate([137 for _ in range(9)]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_normal_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        # self.enc = nn.Linear(9, emb_dim)

    def forward(self, batch):
        encoded_features = 0
        for i in range(batch.x.shape[1]):
            encoded_features += self.atom_embedding_list[i](batch.x[:, i])

        batch.x = encoded_features

        # batch.x = self.enc(batch.x.float())
        return batch


class SimpleBondEncoder(nn.Module):
    """
    The bond Encoder used in Moleculenet dataset.

    Args:
        emb_dim (int): Output edge embedding dimension
    """

    def __init__(self, emb_dim):
        super().__init__()

        self.bond_embedding_list = nn.ModuleList()

        for i, dim in enumerate([14, 14, 14]):
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_normal_(emb.weight.data)
            self.bond_embedding_list.append(emb)

        # self.enc = nn.Linear(3, emb_dim)

    def forward(self, batch):
        bond_embedding = 0
        for i in range(batch.edge_attr.shape[1]):
            edge_attr = batch.edge_attr
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        batch.edge_attr = bond_embedding

        # batch.edge_attr = self.enc(batch.edge_attr.float())

        return batch


class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads, act='relu',
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, log_attn_weights=False):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        if act == "gelu":
            self.activation = nn.GELU
        elif act == "relu":
            self.activation = nn.ReLU

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type not in ['Transformer',
                                                          'BiasedTransformer']:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )

        # Local message-passing model.
        self.local_gnn_with_edge_attr = True
        if local_gnn_type == 'None':
            self.local_model = None

        # MPNNs without edge attributes support.
        elif local_gnn_type == "GCN":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.GCNConv(dim_h, dim_h)
        elif local_gnn_type == 'GIN':
            self.local_gnn_with_edge_attr = False
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h))
            self.local_model = pygnn.GINConv(gin_nn)

        # MPNNs supporting also edge attributes.
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=min(128, dim_h),
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             act=act,
                                             equivstable_pe=equivstable_pe)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type in ['Transformer', 'BiasedTransformer']:
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
            # self.global_model = torch.nn.TransformerEncoderLayer(
            #     d_model=dim_h, nhead=num_heads,
            #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
            #     layer_norm_eps=1e-5, batch_first=True)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout, causal=False)
        elif global_model_type == "BigBird":
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
            # self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == 'CustomGatedGCN':
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(Batch(batch=batch,
                                                   x=h,
                                                   edge_index=batch.edge_index,
                                                   edge_attr=batch.edge_attr,
                                                   pe_EquivStableLapPE=es_data))
                # GatedGCN does residual connection and dropout internally.
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.local_gnn_with_edge_attr:
                    if self.equivstable_pe:
                        h_local = self.local_model(h,
                                                   batch.edge_index,
                                                   batch.edge_attr,
                                                   batch.pe_EquivStableLapPE)
                    else:
                        h_local = self.local_model(h,
                                                   batch.edge_index,
                                                   batch.edge_attr)
                else:
                    h_local = self.local_model(h, batch.edge_index)
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            h_dense, mask = to_dense_batch(h, batch.batch)
            if self.global_model_type == 'Transformer':
                h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            elif self.global_model_type == 'BiasedTransformer':
                # Use Graphormer-like conditioning, requires `batch.attn_bias`.
                h_attn = self._sa_block(h_dense, batch.attn_bias, ~mask)[mask]
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            elif self.global_model_type == 'BigBird':
                h_attn = self.self_attn(h_dense, attention_mask=mask)
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        if not self.log_attn_weights:

            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s


class PrefixGPSLayer(GPSLayer):
    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads, pre_seq_len=10, act='relu',
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, log_attn_weights=False):
        super().__init__(dim_h, local_gnn_type, global_model_type, num_heads, act=act, pna_degrees=pna_degrees,
                         equivstable_pe=equivstable_pe, dropout=dropout, attn_dropout=attn_dropout,
                         layer_norm=layer_norm, batch_norm=batch_norm, bigbird_cfg=bigbird_cfg,
                         log_attn_weights=log_attn_weights
                         )

        for param in self.parameters():
            param.requires_grad = False

        self.prefix_dropout = nn.Dropout(dropout)
        self.pre_seq_len = pre_seq_len
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = GraphGPSPrefixEncoder()

    def get_prompt(self, batch_size, device):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        prompts = self.prefix_encoder(prefix_tokens)
        prompts = prompts.view(batch_size, self.pre_seq_len, self.dim_h)
        prompts = self.prefix_dropout(prompts)
        return prompts

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        batch_size = x.shape[0]
        device = x.device
        prompts = self.get_prompt(batch_size, device)

        x = torch.cat((prompts, x), dim=1)

        if attn_mask is not None:
            prefix_attention_mask = torch.ones(self.pre_seq_len, batch_size).to(device)
            attn_mask = torch.cat((prefix_attention_mask, attn_mask), dim=1)

        if key_padding_mask is not None:
            prefix_attention_padding_mask = torch.zeros(batch_size, self.pre_seq_len)
            prefix_attention_padding_mask = prefix_attention_padding_mask.type(torch.bool).to(device)
            key_padding_mask = torch.cat((prefix_attention_padding_mask, key_padding_mask), dim=1)

        if not self.log_attn_weights:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        x = x[:, self.pre_seq_len:, :]
        return x


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """

    def __init__(self, emb_dim):
        super(FeatureEncoder, self).__init__()
        self.emb_dim = emb_dim

        self.node_encoder = SimpleAtomEncoder(self.emb_dim)

        self.edge_encoder = SimpleBondEncoder(self.emb_dim)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class GPSPretrained(PreTrainedModel):
    config_class = GraphGPSConfig
    base_model_prefix = "graphgps"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GPSModel(GPSPretrained):
    """General-Powerful-Scalable graph transformer.
    https://arxiv.org/abs/2205.12454
    Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D.
    Recipe for a general, powerful, scalable graph transformer. (NeurIPS 2022)
    """

    def __init__(self, config):
        super(GPSModel, self).__init__(config)

        local_gnn_type, global_model_type = config.gt_layer_type.split('+')
        layers = []
        for _ in range(config.gt_layers):
            layers.append(GPSLayer(
                dim_h=config.gt_dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=config.gt_n_heads,
                act=config.gnn_act,
                pna_degrees=0,
                equivstable_pe=False,
                dropout=config.gt_dropout,
                attn_dropout=config.gt_attn_dropout,
                layer_norm=config.gt_layer_norm,
                batch_norm=config.gt_batch_norm,
                bigbird_cfg=None,
                log_attn_weights=False,
            ))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class GPSForGraphPrediction(GPSPretrained):
    def __init__(self, config):
        super(GPSForGraphPrediction, self).__init__(config)
        self.n_tasks = config.n_tasks
        self.task_type = config.task_type

        self.hidden_dim = config.gt_dim_hidden
        self.encoder = FeatureEncoder(self.hidden_dim)

        self.model = GPSModel(config)

        self.post_mp = SANGraphHead(dim_in=config.gnn_dim_inner, dim_out=config.n_tasks)

    def compute_loss(self, pred, true):
        """
        Compute loss and prediction score

        Args:
            pred (torch.tensor): Unnormalized prediction
            true (torch.tensor): Grou

        Returns: Loss, normalized prediction score

        """
        bce_loss = nn.BCEWithLogitsLoss(reduction="sum")
        mse_loss = nn.MSELoss(reduction="sum")

        # default manipulation for pred and true
        # can be skipped if special loss computation is needed
        pred = pred.squeeze(-1) if pred.ndim > 1 else pred
        true = true.squeeze(-1) if true.ndim > 1 else true

        loss = None
        if self.task_type == "regression":
            true = true.float()
            loss = mse_loss(pred, true)
        else:
            loss = bce_loss(pred, true)

        return loss

    def forward(self, batch, return_dict: Optional[bool] = None):
        for module in self.children():
            batch = module(batch)

        outputs, labels = batch
        loss = self.compute_loss(outputs, labels)
        if not return_dict:
            return tuple(x for x in [loss, outputs, labels] if x is not None)
        return SequenceClassifierOutput(loss=loss, logits=outputs, hidden_states=None, attentions=None)


class PrefixGPSForGraphPrediction(GPSForGraphPrediction):
    pass
