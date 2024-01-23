import os
from typing import Optional, Union

import torch
from torch import nn
from transformers import GraphormerForGraphClassification
from .prefix_encoder import GraphPrefixEncoder
from transformers.models.graphormer.modeling_graphormer import GraphormerGraphEncoderLayer, LayerDropModuleList


class GraphormerForGraphPrediction(GraphormerForGraphClassification):
    def __init__(self, config):
        super().__init__(config)
        if config.light_weight_tuning:
            for param in self.encoder.graph_encoder.parameters():
                param.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        config = kwargs.pop('config', None)
        revision = kwargs.pop('revision', None)
        ign = kwargs.pop('ignore_mismatched_sizes', True)
        return super().from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, config=config,
                                       ignore_mismatched_sizes=ign,
                                       revision=revision,
                                       )


class GraphormerGraphEncoderLayerPrefix(GraphormerGraphEncoderLayer):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.config = config
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        for param in self.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.p = torch.nn.Parameter(torch.rand(self.embedding_dim))
        self.prefix_encoder = GraphPrefixEncoder(config)

    def get_prompt(self, batch_size, device):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        prompts = self.prefix_encoder(prefix_tokens)
        prompts = prompts.view(batch_size, self.pre_seq_len, self.embedding_dim)
        prompts = self.dropout(prompts)
        prompts = torch.permute(prompts, (1, 0, 2))
        return prompts

    def forward(
            self,
            input_nodes: torch.Tensor,
            self_attn_bias=None,
            self_attn_mask=None,
            self_attn_padding_mask=None,
    ):
        """
        nn.LayerNorm is applied either before or after the self-attention/ffn modules similar to the original
        Transformer implementation.
        """

        device = input_nodes.device
        batch_size = input_nodes.shape[1]
        prompts = self.get_prompt(batch_size, device)

        input_nodes = input_nodes + self.p
        input_nodes = torch.cat((prompts, input_nodes), dim=0)

        if self_attn_mask is not None:
            prefix_attention_mask = torch.ones(self.pre_seq_len, batch_size).to(device)
            self_attn_mask = torch.cat((prefix_attention_mask, self_attn_mask), dim=0)

        if self_attn_padding_mask is not None:
            prefix_attention_padding_mask = torch.zeros(batch_size, self.pre_seq_len)
            prefix_attention_padding_mask = prefix_attention_padding_mask.type(torch.ByteTensor).to(device)
            self_attn_padding_mask = torch.cat((prefix_attention_padding_mask, self_attn_padding_mask), dim=1)

        if self_attn_bias is not None:
            self_attn_bias = torch.nn.functional.pad(self_attn_bias,
                                                     pad=(self.pre_seq_len, 0, self.pre_seq_len, 0))

        residual = input_nodes
        if self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        input_nodes, attn = self.self_attn(
            query=input_nodes,
            key=input_nodes,
            value=input_nodes,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = residual + input_nodes
        if not self.pre_layernorm:
            input_nodes = self.self_attn_layer_norm(input_nodes)

        residual = input_nodes
        if self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)
        input_nodes = self.activation_fn(self.fc1(input_nodes))
        input_nodes = self.activation_dropout_module(input_nodes)
        input_nodes = self.fc2(input_nodes)
        input_nodes = self.dropout_module(input_nodes)
        input_nodes = residual + input_nodes
        if not self.pre_layernorm:
            input_nodes = self.final_layer_norm(input_nodes)

        input_nodes = input_nodes[self.pre_seq_len:, :, :]
        return input_nodes, attn


class PrefixGraphormerForGraphPrediction(GraphormerForGraphClassification):
    def __init__(self, config):
        super().__init__(config)

        for param in self.encoder.graph_encoder.parameters():
            param.requires_grad = False

        self.config = config
        if self.encoder.graph_encoder.layerdrop > 0.0:
            self.encoder.graph_encoder.layers = LayerDropModuleList(p=self.encoder.graph_encoder.layerdrop)
        else:
            self.encoder.graph_encoder.layers = nn.ModuleList([])

        if config.prefix_layer_max == -1:
            config.prefix_layer_max = config.num_hidden_layers

        self.encoder.graph_encoder.layers.extend([GraphormerGraphEncoderLayer(config) for _ in range(config.prefix_layer_min)] +
                                                 [GraphormerGraphEncoderLayerPrefix(config) for _
                                                  in range(config.prefix_layer_min, config.prefix_layer_max)] +
                                                 [GraphormerGraphEncoderLayer(config) for _ in
                                                  range(config.prefix_layer_max,
                                                        config.num_hidden_layers)])
        for layer in self.encoder.graph_encoder.layers:
            if not isinstance(layer, GraphormerGraphEncoderLayerPrefix):
                for param in layer.parameters():
                    param.requires_grad = False


        for layer in range(config.num_trans_layers_to_freeze):
            m = self.encoder.graph_encoder.layers[layer]
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        graphormer_params = 0
        for name, param in self.encoder.named_parameters():
            if not param.requires_grad:
                graphormer_params += param.numel()

        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - graphormer_params
        print('total param is {}'.format(total_param))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        config = kwargs.pop('config', None)
        revision = kwargs.pop('revision', None)
        ign = kwargs.pop('ignore_mismatched_sizes', True)
        return super().from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, config=config,
                                       ignore_mismatched_sizes=ign,
                                       revision=revision
                                       )
