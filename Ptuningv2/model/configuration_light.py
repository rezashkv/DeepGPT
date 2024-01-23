from torch import nn
from transformers import PretrainedConfig

from transformers.utils import logging
from utils.light_featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES


logger = logging.get_logger(__name__)
vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)

class LiGhTConfig(PretrainedConfig):
    r"""
        This is the configuration class to store the configuration of a [`~LiGHT`]. It is used to instantiate a
        LiGhT model according to the specified arguments, defining the model architecture. Instantiating a
        configuration with the defaults will yield a similar configuration to the base configuration in the paper.

        Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
        documentation from [`PretrainedConfig`] for more information.

        Args:
            d_node_feats : int, optional (default=137)
                The dimension of the node features.
            d_edge_feats : int, optional (default=14)
                The dimension of the edge features.
            d_g_feats : int, optional (default=768)
                The dimension of the graph features.
            d_hpath_ratio : int, optional (default=12)
                The ratio of d_g_feats to calculate the dimension of the hidden state in the GNN.
            n_mol_layers : int, optional (default=12)
                The number of layers in the molecular transformer.
            path_length : int, optional (default=5)
                The length of the shortest path used for constructing the graph embeddings.
            n_heads : int, optional (default=12)
                The number of attention heads in the multi-head attention layer.
            n_ffn_dense_layers : int, optional (default=2)
                The number of dense layers in the feed-forward network layer.
            input_drop : float, optional (default=0.0)
                The dropout probability for the input layer.
            attn_drop : float, optional (default=0.1)
                The dropout probability for the attention layer.
            feat_drop : float, optional (default=0.1)
                The dropout probability for the feed-forward network layer.
            batch_size : int, optional (default=1024)
                The batch size to be used during training.
            lr : float, optional (default=2e-04)
                The learning rate for the optimizer.
            weight_decay : float, optional (default=1e-6)
                The weight decay factor for the optimizer.
            candi_rate : float, optional (default=0.5)
                The rate at which the candidate nodes are used during training.
            fp_disturb_rate : float, optional (default=0.5)
                The rate at which the molecule fingerprints are disturbed during training.
            md_disturb_rate : float, optional (default=0.5)
                The rate at which the molecular descriptors are disturbed during training.
    """

    model_type = "light"

    def __init__(self,
                 d_node_feats=137,
                 d_edge_feats=14,
                 d_g_feats=768,
                 d_hpath_ratio=12,
                 n_mol_layers=12,
                 path_length=5,
                 n_heads=12,
                 n_ffn_dense_layers=2,
                 input_drop=0.0,
                 attn_drop=0.1,
                 feat_drop=0.1,
                 batch_size=1024,
                 lr=2e-04,
                 weight_decay=1e-6,
                 candi_rate=0.5,
                 fp_disturb_rate=0.5,
                 md_disturb_rate=0.5,
                 activation="gelu",
                 d_fp_feats=512,
                 d_md_feats=200,
                 n_node_types=vocab.vocab_size,
                 readout_mode='mean',
                 **kwargs):
        super().__init__(**kwargs)
        self.d_node_feats = d_node_feats
        self.d_edge_feats = d_edge_feats
        self.d_g_feats = d_g_feats
        self.hidden_size = d_g_feats
        self.d_hpath_ratio = d_hpath_ratio
        self.n_mol_layers = n_mol_layers
        self.path_length = path_length
        self.n_heads = n_heads
        self.n_ffn_dense_layers = n_ffn_dense_layers
        self.input_drop = input_drop
        self.attn_drop = attn_drop
        self.feat_drop = feat_drop
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.candi_rate = candi_rate
        self.fp_disturb_rate = fp_disturb_rate
        self.md_disturb_rate = md_disturb_rate
        self.activation = activation
        self.d_fp_feats = d_fp_feats
        self.d_md_feats = d_md_feats
        self.n_node_types = n_node_types
        self.readout_mode = readout_mode
