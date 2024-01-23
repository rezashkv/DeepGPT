import logging
from transformers import (
    GraphormerConfig
)

from model.utils import get_model, TaskType
from tasks.graph.dataset import GraphDataset
from training.trainer_base import BaseTrainer

logger = logging.getLogger(__name__)

def get_trainer(args):
    model_args, data_args, training_args, _ = args

    dataset = GraphDataset(data_args, training_args)

    config = GraphormerConfig.from_pretrained(
        model_args.model_name_or_path,
        finetuning_task=data_args.dataset_name,
        revision=model_args.model_revision,
        num_classes=data_args.n_tasks
    )
    config.light_weight_tuning = model_args.lightweight
    config.prefix_layer_min = model_args.prefix_layer_min
    config.prefix_layer_max = model_args.prefix_layer_max


    model = get_model(model_args, TaskType.GRAPH, config)

    # Initialize our Trainer
    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        data_collator=dataset.data_collator,
        test_key=dataset.metric
    )

    return trainer, dataset