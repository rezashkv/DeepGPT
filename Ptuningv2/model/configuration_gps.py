from transformers import PretrainedConfig


class GraphGPSConfig(PretrainedConfig):
    """
        Configuration class for GraphGPS.

        Args:
            out_dir (str, optional):
                Output directory path.
            model_type (str, optional):
                Type of the model.
            model_loss_fun (str, optional):
                Loss function used in the model.
            model_edge_decoding (str, optional):
                Edge decoding method.
            model_graph_pooling (str, optional):
                Graph pooling method.
            gt_layer_type (str, optional):
                Type of the layer used in the CustomGatedGCN+Transformer.
            gt_layers (int, optional):
                Number of layers in the CustomGatedGCN+Transformer.
            gt_n_heads (int, optional):
                Number of attention heads in the CustomGatedGCN+Transformer.
            gt_dim_hidden (int, optional):
                Dimension of the hidden layer in the CustomGatedGCN+Transformer.
            gt_dropout (float, optional):
                Dropout rate in the CustomGatedGCN+Transformer.
            gt_attn_dropout (float, optional):
                Attention dropout rate in the CustomGatedGCN+Transformer.
            gt_layer_norm (bool, optional):
                Whether to apply layer normalization in the CustomGatedGCN+Transformer.
            gt_batch_norm (bool, optional):
                Whether to apply batch normalization in the CustomGatedGCN+Transformer.
            prefix_enable (bool, optional):
                Whether to enable prefix in the model.
            prefix_projection (bool, optional):
                Whether to enable prefix projection.
            prefix_pre_seq_len (int, optional):
                Length of the prefix sequence.
            gnn_head (str, optional):
                GNN head type.
            gnn_layers_pre_mp (int, optional):
                Number of GNN layers before message passing.
            gnn_layers_post_mp (int, optional):
                Number of GNN layers after message passing.
            gnn_dim_inner (int, optional):
                Dimension of the inner layer in GNN.
            gnn_batchnorm (bool, optional):
                Whether to apply batch normalization in GNN.
            gnn_act (str, optional):
                Activation function in GNN.
            gnn_dropout (float, optional):
                Dropout rate in GNN.
            gnn_agg (str, optional):
                Aggregation method in GNN.
            gnn_normalize_adj (bool, optional):
                Whether to normalize adjacency matrix in GNN.
            optim_clip_grad_norm (bool, optional):
                Whether to clip the gradient norm during optimization.
            optim_optimizer (str, optional):
                Optimizer used in training.
            optim_weight_decay (float, optional):
                Weight decay value for regularization.
            optim_base_lr (float, optional):
                Base learning rate.
            optim_max_epoch (int, optional):
                Maximum number of training epochs.
            optim_scheduler (str, optional):
                Learning rate scheduler.
            optim_num_warmup_epochs (int, optional):
                Number of warmup epochs for the scheduler.
            **kwargs (optional):
                Additional keyword arguments.

        """

    model_type = "gps"

    def __init__(
            self,
            out_dir="/home/reza/research/results/GraphGPS/gps-bbbp-finetune",
            model_loss_fun="cross_entropy",
            model_edge_decoding="dot",
            model_graph_pooling="mean",
            gt_layer_type="CustomGatedGCN+Performer",
            gt_layers=16,
            gt_n_heads=8,
            gt_dim_hidden=256,
            gt_dropout=0.1,
            gt_attn_dropout=0.1,
            gt_layer_norm=False,
            gt_batch_norm=True,
            prefix_enable=False,
            prefix_projection=False,
            prefix_pre_seq_len=10,
            gnn_head="san_graph",
            gnn_layers_pre_mp=0,
            gnn_layers_post_mp=3,
            gnn_dim_inner=256,
            gnn_batchnorm=True,
            gnn_act="gelu",
            gnn_dropout=0.0,
            gnn_agg="mean",
            gnn_normalize_adj=False,
            optim_clip_grad_norm=True,
            optim_optimizer="adamW",
            optim_weight_decay=1e-5,
            optim_base_lr=0.001,
            optim_max_epoch=100,
            optim_scheduler="cosine_with_warmup",
            optim_num_warmup_epochs=5,
            **kwargs
    ):
        super(GraphGPSConfig, self).__init__(**kwargs)
        self.out_dir = out_dir
        self.model_loss_fun = model_loss_fun
        self.model_edge_decoding = model_edge_decoding
        self.model_graph_pooling = model_graph_pooling
        self.gt_layer_type = gt_layer_type
        self.gt_layers = gt_layers
        self.gt_n_heads = gt_n_heads
        self.gt_dim_hidden = gt_dim_hidden
        self.gt_dropout = gt_dropout
        self.gt_attn_dropout = gt_attn_dropout
        self.gt_layer_norm = gt_layer_norm
        self.gt_batch_norm = gt_batch_norm
        self.prefix_enable = prefix_enable
        self.prefix_projection = prefix_projection
        self.prefix_pre_seq_len = prefix_pre_seq_len
        self.gnn_head = gnn_head
        self.gnn_layers_pre_mp = gnn_layers_pre_mp
        self.gnn_layers_post_mp = gnn_layers_post_mp
        self.gnn_dim_inner = gnn_dim_inner
        self.gnn_batchnorm = gnn_batchnorm
        self.gnn_act = gnn_act
        self.gnn_dropout = gnn_dropout
        self.gnn_agg = gnn_agg
        self.gnn_normalize_adj = gnn_normalize_adj
        self.optim_clip_grad_norm = optim_clip_grad_norm
        self.optim_optimizer = optim_optimizer
        self.optim_weight_decay = optim_weight_decay
        self.optim_base_lr = optim_base_lr
        self.optim_max_epoch = optim_max_epoch
        self.optim_scheduler = optim_scheduler
        self.optim_num_warmup_epochs = optim_num_warmup_epochs
