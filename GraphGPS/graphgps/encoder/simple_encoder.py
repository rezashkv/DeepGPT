import torch
from torch import nn
from torch_geometric.graphgym.register import register_edge_encoder, register_node_encoder

@register_node_encoder('SimpleAtom')
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

@register_edge_encoder('SimpleBond')
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