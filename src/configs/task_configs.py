from train.train_vqa import train_vqa
from train.train_nlvr2 import train_nlvr2
from train.train_snli_ve import train_snli_ve

SUPPORTED_VL_TASKS = ['vqa', 'nlvr2', 'snli-ve']

mscoco_config = {
    'data_dir': '/data/datasets/MCL/ms-coco'
}

flickr_config = {
    'data_dir': '/data/datasets/MCL/flickr30k'
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
        'train_method': train_vqa
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
        'train_method': train_nlvr2
}

snli_ve_config = {
        'task_name': 'SNLI-VE',
        'data_dir': '/data/datasets/MCL/snli-ve',
        'images_source': 'flickr30k',
        'splits': ['train', 'dev', 'test'],
        'num_labels': 3,
        'model_type': 'classification',
        'num_epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
        'train_method': train_snli_ve
}

task_configs = {
    'ms-coco': mscoco_config,
    'flickr30k': flickr_config,
    'vqa': vqa_config,
    'nlvr2': nlvr_config,
    'snli-ve': snli_ve_config
}
