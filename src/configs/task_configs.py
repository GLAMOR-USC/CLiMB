from train.train_vqa import train_vqa, eval_vqa_forgetting
from train.train_nlvr2 import train_nlvr2, eval_nlvr2_forgetting
from train.train_snli_ve import train_snli_ve, eval_snli_ve_forgetting

from train.train_mscoco_detection import train_mscoco_detection

SUPPORTED_VL_TASKS = ['vqa', 'nlvr2', 'snli-ve']

mscoco_config = {
        'data_dir': '/data/datasets/MCL/ms-coco',
}

mscoco_detection_config = {
        'task_name': 'MLIC',
        'annotation_dir': '/data/datasets/MCL/ms-coco/detections/annotations/',
        'images_source': 'ms-coco',
        'splits': ['train', 'val'],
        'num_labels': 80,
        'model_type': 'classification',
        'num_epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'train_method': train_mscoco_detection
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
        'train_method': train_vqa,
        'eval_forgetting_method': eval_vqa_forgetting
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
        'eval_forgetting_method': eval_nlvr2_forgetting
}

snli_ve_config = {
        'task_name': 'SNLI-VE',
        'data_dir': '/data/datasets/MCL/snli-ve',
        'images_source': 'flickr30k',
        'splits': ['train', 'dev', 'test'],
        'num_labels': 3,
        'model_type': 'classification',
        'num_epochs': 5,
        'lr': 5e-5,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
        'train_method': train_snli_ve,
        'eval_forgetting_method': eval_snli_ve_forgetting
}

task_configs = {
    'ms-coco': mscoco_config,
    'flickr30k': flickr_config,
    'vqa': vqa_config,
    'nlvr2': nlvr_config,
    'ms-coco_detection': mscoco_detection_config,
    'snli-ve': snli_ve_config
}
