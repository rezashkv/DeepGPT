from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_prefix')
def set_cfg_pretrained(cfg):
    """Configuration options for applying graph prompt tuning.
    """

    cfg.prefix = CN()

    # user prefix graph prompt tuning
    cfg.prefix.enable = False

    # project prompt tokens
    cfg.prefix.prefix_projection = False

    # Freeze the main pretrained 'body' of the model, learning only the new head
    cfg.prefix.pre_seq_len = 0

    cfg.prefix.min = 0
    cfg.prefix.max = -1


@register_config('cfg_lightweight')
def set_cfg_pretrained(cfg):
    """Configuration options for applying lightweight fine-tuning.
    """

    cfg.lightweight = CN()

    # user prefix graph prompt tuning
    cfg.lightweight.enable = False
    cfg.lightweight.num_freeze_layers = 0
