from train.train_vqa import train_vqa, eval_vqa_forgetting, vqa_replay_step
from train.train_nlvr2 import train_nlvr2, eval_nlvr2_forgetting, nlvr2_replay_step
from train.train_snli_ve import train_snli_ve, eval_snli_ve_forgetting, snli_ve_replay_step

from data.visionlanguage_datasets.vqa_dataset import vqa_batch_collate
from data.visionlanguage_datasets.nlvr2_dataset import nlvr2_batch_collate
from data.visionlanguage_datasets.snli_ve_dataset import snlive_batch_collate

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
        'num_images': 1,
        'model_type': 'classification',
        'num_epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
        'train_method': train_vqa,
        'eval_forgetting_method': eval_vqa_forgetting,
        'batch_collate_fn': vqa_batch_collate,
        'replay_step_method': vqa_replay_step
}

nlvr_config = {
        'task_name': 'NLVRv2',
        'data_dir': '/data/datasets/MCL/nlvr2',
        'splits': ['train', 'val'],
        'num_labels': 2,
        'num_images': 2,
        'model_type': 'classification',
        'num_epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
        'train_method': train_nlvr2,
        'eval_forgetting_method': eval_nlvr2_forgetting,
        'batch_collate_fn': nlvr2_batch_collate,
        'replay_step_method': nlvr2_replay_step
}

snli_ve_config = {
        'task_name': 'SNLI-VE',
        'data_dir': '/data/datasets/MCL/snli-ve',
        'images_source': 'flickr30k',
        'splits': ['train', 'dev', 'test'],
        'num_labels': 3,
        'num_images': 1,
        'model_type': 'classification',
        'num_epochs': 5,
        'lr': 5e-5,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
        'train_method': train_snli_ve,
        'eval_forgetting_method': eval_snli_ve_forgetting,
        'batch_collate_fn': snlive_batch_collate,
        'replay_step_method': snli_ve_replay_step
}

task_configs = {
    'ms-coco': mscoco_config,
    'flickr30k': flickr_config,
    'vqa': vqa_config,
    'nlvr2': nlvr_config,
    'snli-ve': snli_ve_config
}
