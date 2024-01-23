from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('pe_simple')
def set_cfg_pretrained(cfg):
    """Configuration options for applying simple linear positional encoding
    """

    cfg.pe_simple = CN()

    # user prefix graph prompt tuning
    cfg.pe_simple.enable = False

    cfg.pe_simple.atom_feats = 9
    cfg.pe_simple.bond_feats = 14
