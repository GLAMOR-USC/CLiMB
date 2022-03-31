from train.train_vqa import train_vqa, eval_vqa_forgetting
from train.train_nlvr2 import train_nlvr2, eval_nlvr2_forgetting

from data.visionlanguage_datasets.vqa_dataset import vqa_batch_collate
from data.visionlanguage_datasets.nlvr2_dataset import nlvr2_batch_collate

SUPPORTED_VL_TASKS = ['vqa', 'nlvr2']

mscoco_config = {
    'data_dir': '/data/datasets/MCL/ms-coco'
}

vqa_config = {
        'task_name': 'VQAv2',
        'data_dir': '/data/datasets/MCL/vqav2',
        'images_source': 'ms-coco',
        'splits': ['train', 'val'],
        'num_labels': 3129,
        'model_type': 'classification',
        'num_epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
        'train_method': train_vqa,
        'eval_forgetting_method': eval_vqa_forgetting,
        'batch_collate_fn': vqa_batch_collate
}

nlvr_config = {
        'task_name': 'NLVRv2',
        'data_dir': '/data/datasets/MCL/nlvr2',
        'splits': ['train', 'val'],
        'num_labels': 2,
        'model_type': 'classification',
        'num_epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
        'train_method': train_nlvr2,
        'eval_forgetting_method': eval_nlvr2_forgetting,
        'batch_collate_fn': nlvr2_batch_collate
}

task_configs = {
    'ms-coco': mscoco_config,
    'vqa': vqa_config,
    'nlvr2': nlvr_config,
}
