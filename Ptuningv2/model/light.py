import os

import dgl
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from torch.nn import MSELoss, BCEWithLogitsLoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .configuration_light import LiGhTConfig
from typing import Optional, Union
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from tasks.light.featurizer import VIRTUAL_ATOM_FEATURE_PLACEHOLDER, VIRTUAL_BOND_FEATURE_PLACEHOLDER

from .prefix_encoder import GraphPrefixEncoder


class Residual(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_ffn_dense_layers, feat_drop, activation):
        super(Residual, self).__init__()
        self.norm = nn.LayerNorm(d_in_feats)
        self.in_proj = nn.Linear(d_in_feats, d_out_feats)
        self.ffn = MLP(d_out_feats, d_out_feats, n_ffn_dense_layers, activation, d_hidden_feats=d_out_feats * 4)
        self.feat_dropout = nn.Dropout(feat_drop)

    def forward(self, x, y):
        x = x + self.feat_dropout(self.in_proj(y))
        y = self.norm(x)
        y = self.ffn(y)
        y = self.feat_dropout(y)
        x = x + y
        return x


class MLP(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_dense_layers, activation, d_hidden_feats=None):
        super(MLP, self).__init__()
        self.n_dense_layers = n_dense_layers
        self.d_hidden_feats = d_out_feats if d_hidden_feats is None else d_hidden_feats
        self.dense_layer_list = nn.ModuleList()
        self.in_proj = nn.Linear(d_in_feats, self.d_hidden_feats)
        for _ in range(self.n_dense_layers - 2):
            self.dense_layer_list.append(nn.Linear(self.d_hidden_feats, self.d_hidden_feats))
        self.out_proj = nn.Linear(self.d_hidden_feats, d_out_feats)
        self.act = getattr(nn, activation.upper())()

    def forward(self, feats):
        feats = self.act(self.in_proj(feats))
        for i in range(self.n_dense_layers - 2):
            feats = self.act(self.dense_layer_list[i](feats))
        feats = self.out_proj(feats)
        return feats


class TripletTransformer(nn.Module):
    def __init__(self,
                 d_feats,
                 d_hpath_ratio,
                 path_length,
                 n_heads,
                 n_ffn_dense_layers,
                 feat_drop=0.,
                 attn_drop=0.,
                 activation="gelu"):
        super(TripletTransformer, self).__init__()
        self.d_feats = d_feats
        self.d_trip_path = d_feats // d_hpath_ratio
        self.path_length = path_length
        self.n_heads = n_heads
        self.scale = d_feats ** (-0.5)

        self.attention_norm = nn.LayerNorm(d_feats)
        self.qkv = nn.Linear(d_feats, d_feats * 3)
        self.node_out_layer = Residual(d_feats, d_feats, n_ffn_dense_layers, feat_drop, activation)

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.act = getattr(nn, activation.upper())()

    def pretrans_edges(self, edges):
        edge_h = edges.src['hv']
        return {"he": edge_h}

    def forward(self, g, triplet_h, dist_attn, path_attn):
        g = g.local_var()
        new_triplet_h = self.attention_norm(triplet_h)
        qkv = self.qkv(new_triplet_h).reshape(-1, 3, self.n_heads, self.d_feats // self.n_heads).permute(1, 0, 2, 3)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        g.dstdata.update({'K': k})
        g.srcdata.update({'Q': q})
        g.apply_edges(fn.u_dot_v('Q', 'K', 'node_attn'))

        g.edata['a'] = g.edata['node_attn'] + dist_attn.reshape(len(g.edata['node_attn']), -1, 1) + path_attn.reshape(
            len(g.edata['node_attn']), -1, 1)
        g.edata['sa'] = self.attn_dropout(edge_softmax(g, g.edata['a']))

        g.ndata['hv'] = v.view(-1, self.d_feats)
        g.apply_edges(self.pretrans_edges)
        g.edata['he'] = ((g.edata['he'].view(-1, self.n_heads, self.d_feats // self.n_heads)) * g.edata['sa']).view(-1,
                                                                                                                    self.d_feats)

        g.update_all(fn.copy_e('he', 'm'), fn.sum('m', 'agg_h'))
        return self.node_out_layer(triplet_h, g.ndata['agg_h'])

    def _device(self):
        return next(self.parameters()).device


class AtomEmbedding(nn.Module):
    def __init__(
            self,
            d_atom_feats,
            d_g_feats,
            input_drop):
        super(AtomEmbedding, self).__init__()
        self.in_proj = nn.Linear(d_atom_feats, d_g_feats)
        self.virtual_atom_emb = nn.Embedding(1, d_g_feats)
        self.input_dropout = nn.Dropout(input_drop)

    def forward(self, pair_node_feats, indicators):
        pair_node_h = self.in_proj(pair_node_feats)
        pair_node_h[indicators == VIRTUAL_ATOM_FEATURE_PLACEHOLDER, 1, :] = self.virtual_atom_emb.weight  # .half()
        return torch.sum(self.input_dropout(pair_node_h), dim=-2)


class BondEmbedding(nn.Module):
    def __init__(
            self,
            d_bond_feats,
            d_g_feats,
            input_drop):
        super(BondEmbedding, self).__init__()
        self.in_proj = nn.Linear(d_bond_feats, d_g_feats)
        self.virutal_bond_emb = nn.Embedding(1, d_g_feats)
        self.input_dropout = nn.Dropout(input_drop)

    def forward(self, edge_feats, indicators):
        edge_h = self.in_proj(edge_feats)
        edge_h[indicators == VIRTUAL_BOND_FEATURE_PLACEHOLDER] = self.virutal_bond_emb.weight  # .half()
        return self.input_dropout(edge_h)


class TripletEmbedding(nn.Module):
    def __init__(
            self,
            d_g_feats,
            d_fp_feats,
            d_md_feats,
            activation="gelu"):
        super(TripletEmbedding, self).__init__()
        self.in_proj = MLP(d_g_feats * 2, d_g_feats, 2, activation)
        self.fp_proj = MLP(d_fp_feats, d_g_feats, 2, activation)
        self.md_proj = MLP(d_md_feats, d_g_feats, 2, activation)

    def forward(self, node_h, edge_h, fp, md, indicators):
        triplet_h = torch.cat([node_h, edge_h], dim=-1)
        triplet_h = self.in_proj(triplet_h)
        triplet_h[indicators == 1] = self.fp_proj(fp)
        triplet_h[indicators == 2] = self.md_proj(md)
        return triplet_h


class LiGhTPretrained(PreTrainedModel):
    config_class = LiGhTConfig
    base_model_prefix = "light"

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


class LiGhTModel(LiGhTPretrained):
    def __init__(self, config):
        super(LiGhTModel, self).__init__(config)
        self.n_mol_layers = config.n_mol_layers
        self.n_heads = config.n_heads
        self.path_length = config.path_length
        self.d_g_feats = config.d_g_feats
        self.d_trip_path = config.d_g_feats // config.d_hpath_ratio

        self.mask_emb = nn.Embedding(1, config.d_g_feats)
        # Distance Attention
        self.path_len_emb = nn.Embedding(config.path_length + 1, config.d_g_feats)
        self.virtual_path_emb = nn.Embedding(1, config.d_g_feats)
        self.self_loop_emb = nn.Embedding(1, config.d_g_feats)
        self.activation = getattr(nn, config.activation.upper())()
        self.dist_attn_layer = nn.Sequential(
            nn.Linear(self.d_g_feats, self.d_g_feats),
            self.activation,
            nn.Linear(self.d_g_feats, self.n_heads)
        )
        # Path Attention
        self.trip_fortrans = nn.ModuleList([
            MLP(self.d_g_feats, self.d_trip_path, 2, config.activation) for _ in range(self.path_length)
        ])
        self.path_attn_layer = nn.Sequential(
            nn.Linear(self.d_trip_path, self.d_trip_path),
            self.activation,
            nn.Linear(self.d_trip_path, self.n_heads)
        )
        # Molecule Transformer Layers
        self.mol_T_layers = nn.ModuleList([
            TripletTransformer(config.d_g_feats, config.d_hpath_ratio, config.path_length, config.n_heads,
                               config.n_ffn_dense_layers, config.feat_drop, config.attn_drop,
                               config.activation) for _ in range(config.n_mol_layers)
        ])

        self.feat_dropout = nn.Dropout(p=config.feat_drop)
        self.attn_dropout = nn.Dropout(p=config.attn_drop)
        self.act = self.activation

    def _featurize_path(self, g, path_indices):
        mask = (path_indices[:, :] >= 0).to(torch.int32)
        path_feats = torch.sum(mask, dim=-1)
        path_feats = self.path_len_emb(path_feats)
        path_feats[g.edata['vp'] == 1] = self.virtual_path_emb.weight  # virtual path
        path_feats[g.edata['sl'] == 1] = self.self_loop_emb.weight  # self loop
        return path_feats

    def _init_path(self, g, triplet_h, path_indices):
        g = g.local_var()
        path_indices[path_indices < -99] = -1
        path_h = []
        for i in range(self.path_length):
            path_h.append(torch.cat(
                [self.trip_fortrans[i](triplet_h), torch.zeros(size=(1, self.d_trip_path)).to(self._device())], dim=0)[
                              path_indices[:, i]])
        path_h = torch.stack(path_h, dim=-1)
        mask = (path_indices >= 0).to(torch.int32)
        path_size = torch.sum(mask, dim=-1, keepdim=True)
        path_h = torch.sum(path_h, dim=-1) / path_size
        return path_h

    def forward(self, g, triplet_h):
        path_indices = g.edata['path']
        dist_h = self._featurize_path(g, path_indices)
        path_h = self._init_path(g, triplet_h, path_indices)
        dist_attn, path_attn = self.dist_attn_layer(dist_h), self.path_attn_layer(path_h)
        for i in range(self.n_mol_layers):
            triplet_h = self.mol_T_layers[i](g, triplet_h, dist_attn, path_attn)
        return triplet_h

    def _device(self):
        return next(self.parameters()).device


class LiGhTForGraphPrediction(LiGhTPretrained):
    def __init__(self, config):
        super(LiGhTForGraphPrediction, self).__init__(config)
        self.d_g_feats = config.d_g_feats
        self.readout_mode = config.readout_mode
        self.activation = getattr(nn, config.activation.upper())()
        # Input
        self.node_emb = AtomEmbedding(config.d_node_feats, config.d_g_feats, config.input_drop)
        self.edge_emb = BondEmbedding(config.d_edge_feats, config.d_g_feats, config.input_drop)
        self.triplet_emb = TripletEmbedding(config.d_g_feats, config.d_fp_feats, config.d_md_feats, config.activation)
        self.mask_emb = nn.Embedding(1, config.d_g_feats)

        # Model
        self.model = LiGhTModel(config)

        self.n_tasks = config.n_tasks
        self.task_type = config.task_type

        # Predict
        # self.node_predictor = nn.Linear(d_g_feats, n_node_types)
        # self.node_predictor = nn.Sequential(
        #     nn.Linear(config.d_g_feats, config.d_g_feats),
        #     self.activation,
        #     nn.Linear(config.d_g_feats, config.n_node_types)
        # )
        # self.fp_predictor = nn.Sequential(
        #     nn.Linear(config.d_g_feats, config.d_g_feats),
        #     self.activation,
        #     nn.Linear(config.d_g_feats, config.d_fp_feats)
        # )
        # self.md_predictor = nn.Sequential(
        #     nn.Linear(config.d_g_feats, config.d_g_feats),
        #     self.activation,
        #     nn.Linear(config.d_g_feats, config.d_md_feats)
        # )
        self.predictor = self.get_predictor(n_tasks=self.n_tasks,
                                            n_layers=2,
                                            predictor_drop=config.feat_drop,
                                            d_hidden_feats=256)

        if config.light_weight_tuning:
            for param in self.model.parameters():
                param.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        new_state_dict = {k.replace('module.', ''): v for k, v in torch.load(f'{pretrained_model_name_or_path}').items()}
        config = kwargs.pop('config', None)
        revision = kwargs.pop('revision', None)
        ign = kwargs.pop('ignore_mismatched_sizes', True)
        return super().from_pretrained(pretrained_model_name_or_path=None, config=config,
                                            ignore_mismatched_sizes=ign,
                                            state_dict=new_state_dict,
                                            revision=revision
                                            )
    def get_predictor(self, n_tasks, n_layers, predictor_drop, d_hidden_feats=None):
        d_input_feats = self.d_g_feats * 3
        if n_layers == 1:
            predictor = nn.Linear(d_input_feats, n_tasks)
        else:
            predictor = nn.ModuleList()
            predictor.append(nn.Linear(d_input_feats, d_hidden_feats))
            predictor.append(nn.Dropout(predictor_drop))
            predictor.append(nn.GELU())
            for _ in range(n_layers - 2):
                predictor.append(nn.Linear(d_hidden_feats, d_hidden_feats))
                predictor.append(nn.Dropout(predictor_drop))
                predictor.append(nn.GELU())
            predictor.append(nn.Linear(d_hidden_feats, n_tasks))
            predictor = nn.Sequential(*predictor)
        return predictor.to(self.model.device)

    def generate_fps(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virtual atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        # Model
        triplet_h = self.model(g, triplet_h)
        # Readout
        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]
        g.ndata['ht'] = triplet_h
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)
        return g_feats

    def forward_train(self, g, fp, md):
        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        triplet_h[g.ndata['mask'] == 1] = self.mask_emb.weight
        # Model
        triplet_h = self.model(g, triplet_h)
        # Predict
        return self.node_predictor(triplet_h[g.ndata['mask'] >= 1]), self.fp_predictor(
            triplet_h[indicators == 1]), self.md_predictor(triplet_h[indicators == 2])

    def forward(self,
                g,
                fp,
                md,
                label,
                smiles: Optional[list] = None,
                return_dict: Optional[bool] = None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        fp = fp.to(self.model.device)
        md = md.to(self.model.device)
        g = g.to(self.model.device)
        labels = label.to(self.model.device)

        indicators = g.ndata[
            'vavn']  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        # Model
        triplet_h = self.model(g, triplet_h)
        g.ndata['ht'] = triplet_h
        # Readout
        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]
        g.remove_nodes(np.where(indicators.detach().cpu().numpy() >= 1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)
        # Predict
        logits = self.predictor(g_feats)

        loss = None
        if labels is not None:
            mask = ~torch.isnan(labels)
            if self.task_type == 'regression':
                loss_fct = MSELoss()
                loss = loss_fct(logits[mask].squeeze(), labels[mask].squeeze().float())
            else:  # Binary or multi-task classificaticon
                loss_fct = BCEWithLogitsLoss(reduction="sum")
                loss = loss_fct(logits[mask], labels[mask])

        if not return_dict:
            return tuple(x for x in [loss, logits] if x is not None)
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=None, attentions=None)


class PrefixTripletTransformer(TripletTransformer):
    def __init__(self, config):
        super().__init__(config.d_g_feats, config.d_hpath_ratio, config.path_length, config.n_heads,
                         config.n_ffn_dense_layers, config.feat_drop, config.attn_drop,
                         config.activation)
        self.dropout = torch.nn.Dropout(config.attn_drop)

        for param in self.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.p = torch.nn.Parameter(torch.rand(self.d_feats))
        self.prefix_encoder = GraphPrefixEncoder(config)

    def get_prompt(self, size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(size, -1).to(self._device())
        prompts = self.prefix_encoder(prefix_tokens)
        prompts = prompts.view(size, self.pre_seq_len, self.d_feats)
        prompts = self.dropout(prompts)
        return prompts

    def forward(self, g, triplet_h, dist_attn, path_attn, node_masks):
        g = g.local_var()

        triplet_h = triplet_h + self.p
        prompts = self.get_prompt(g.batch_size)
        prompted_triplet_h = torch.zeros(
            (triplet_h.shape[0] + prompts.shape[0] * prompts.shape[1], triplet_h.shape[1])).to(triplet_h.device)
        m = [i for i in range(len(node_masks)) if node_masks[i] == 1]
        m_not = [i for i in range(len(node_masks)) if node_masks[i] == 0]
        prompted_triplet_h[m, :] = triplet_h
        prompted_triplet_h[m_not, :] = prompts.view(prompts.shape[0] * prompts.shape[1], prompts.shape[2])

        new_triplet_h = self.attention_norm(prompted_triplet_h)
        qkv = self.qkv(new_triplet_h).reshape(-1, 3, self.n_heads, self.d_feats // self.n_heads).permute(1, 0, 2, 3)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        g.dstdata.update({'K': k})
        g.srcdata.update({'Q': q})
        g.apply_edges(fn.u_dot_v('Q', 'K', 'node_attn'))

        g.edata['a'] = g.edata['node_attn'] + dist_attn.reshape(len(g.edata['node_attn']), -1, 1) + \
                       path_attn.reshape(len(g.edata['node_attn']), -1, 1)
        g.edata['sa'] = self.attn_dropout(edge_softmax(g, g.edata['a']))
        g.ndata['hv'] = v.view(-1, self.d_feats)
        g.apply_edges(self.pretrans_edges)
        g.edata['he'] = ((g.edata['he'].view(-1, self.n_heads, self.d_feats // self.n_heads)) * g.edata['sa']).view(-1,
                                                                                                                    self.d_feats)
        g.update_all(fn.copy_e('he', 'm'), fn.sum('m', 'agg_h'))

        return self.node_out_layer(triplet_h, g.ndata['agg_h'][m, :])


class PrefixLiGhTModel(LiGhTModel):
    def __init__(self, config):
        super().__init__(config)
        self.pre_seq_len = config.pre_seq_len
        self.d_feats = self.d_g_feats

        for param in self.parameters():
            param.requires_grad = False

        self.mol_T_layers = nn.ModuleList([
            PrefixTripletTransformer(config) for _ in range(config.n_mol_layers)
        ])

    def forward(self, g, triplet_h):
        # draw the graph
        path_indices = g.edata['path']
        dist_h = self._featurize_path(g, path_indices)
        path_h = self._init_path(g, triplet_h, path_indices)
        dist_attn, path_attn = self.dist_attn_layer(dist_h), self.path_attn_layer(path_h)

        graphs = dgl.unbatch(g)
        # Define the number of new nodes to add
        num_new_nodes = self.pre_seq_len
        # Add new nodes to each graph in the batch
        edge_masks = []
        node_masks = []
        for i in range(g.batch_size):
            g_i = graphs[i]
            n_nodes_i = g_i.num_nodes()
            node_masks.extend([1 for _ in range(n_nodes_i)])
            node_masks.extend([0 for _ in range(num_new_nodes)])
            g_i.add_nodes(num_new_nodes)
            src = torch.arange(n_nodes_i, n_nodes_i + num_new_nodes).repeat(n_nodes_i).to(g.device)
            dst = torch.arange(n_nodes_i).repeat_interleave(num_new_nodes).to(g.device)
            edge_masks.extend([1 for _ in range(g_i.num_edges())])
            edge_masks.extend([0 for _ in range(len(src))])
            g_i.add_edges(src, dst)
            graphs[i] = g_i

        g = dgl.batch(graphs)
        new_dist_attn = torch.zeros((len(edge_masks), dist_attn.shape[1])).to(dist_attn.device)
        ind = 0
        for i, elem in enumerate(edge_masks):
            if elem:
                new_dist_attn[i, :] = dist_attn[ind, :]
                ind += 1

        new_path_attn = torch.zeros((len(edge_masks), path_attn.shape[1])).to(path_attn.device)
        ind = 0
        for i, elem in enumerate(edge_masks):
            if elem:
                new_path_attn[elem, :] = path_attn[ind, :]
                ind += 1

        for i in range(self.n_mol_layers):
            triplet_h = self.mol_T_layers[i](g, triplet_h, new_dist_attn, new_path_attn, node_masks)
        prompt_nodes = [i for i in range(len(node_masks)) if node_masks[i] == 0]
        g.remove_nodes(prompt_nodes)
        return triplet_h


class PrefixLiGhTForGraphPrediction(LiGhTForGraphPrediction):
    def __init__(self, config: LiGhTConfig):
        super().__init__(config)

        self.config = config

        self.model = PrefixLiGhTModel(config)

        self.pre_seq_len = config.pre_seq_len
        self.d_feats = self.d_g_feats

        light_params = 0
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                light_params += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - light_params
        print('total param is {}'.format(total_param))
