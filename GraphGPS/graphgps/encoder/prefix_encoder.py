import torch
from torch_geometric.graphgym.config import cfg


class GraphGPSPrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, hidden)
    '''
    def __init__(self):
        super().__init__()
        self.prefix_projection = cfg.prefix.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(cfg.prefix.pre_seq_len, cfg.gt.dim_hidden)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(cfg.gt.dim_hidden, cfg.gt.dim_hidden),
                torch.nn.Tanh(),
                torch.nn.Linear(cfg.gt.dim_hidden,cfg.gt.dim_hidden)
            )
        else:
            self.embedding = torch.nn.Embedding(cfg.prefix.pre_seq_len, cfg.gt.dim_hidden)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


