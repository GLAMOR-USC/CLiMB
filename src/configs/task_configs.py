from train.train_vqa import train_vqa
from train.train_mscoco import train_mscoco

SUPPORTED_VL_TASKS = ['vqa']

mscoco_config = {
    'task_name': 'MLIC',
        'data_dir': '/data/datasets/MCL/ms-coco',
        'annotation_dir': '/data/datasets/MCL/ms-coco/detections/annotations/',
        'images_source': 'ms-coco',
        'splits': ['train', 'val'],
        'num_labels': 80,
        'model_type': 'classification',
        'num_epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'train_method': train_mscoco
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
        'train_method': train_vqa
}

task_configs = {
    'ms-coco': mscoco_config,
    'vqa': vqa_config
}
