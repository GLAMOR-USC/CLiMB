from train.visionlanguage_tasks.train_vqa import VQATrainer, LowShotVQATrainer
from train.visionlanguage_tasks.train_nlvr2 import NLVR2Trainer, LowShotNLVR2Trainer
from train.visionlanguage_tasks.train_snli_ve import SNLIVETrainer, LowShotSNLIVETrainer
from train.visionlanguage_tasks.train_vcr import VCRTrainer, LowShotVCRTrainer

SUPPORTED_VL_TASKS = ['vqa', 'nlvr2', 'snli-ve', 'vcr']

mscoco_config = {
        'data_dir': 'ms-coco/',
}

flickr_config = {
    'data_dir': 'flickr30k/',
}

vqa_config = {
        'task_name': 'VQAv2',
        'data_dir': 'vqav2/',
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
        'task_trainer': VQATrainer,
        'random_baseline_score': 0.0,
        'low_shot_config': {'task_trainer': LowShotVQATrainer,
                            'type': 'percentage',
                            'percentage':0.05,
                            'eval_epochs': [6, 8, 10]}
}

nlvr_config = {
        'task_name': 'NLVRv2',
        'data_dir': 'nlvr2/',
        'splits': ['train', 'val'],
        'num_labels': 2,
        'num_images': 2,
        'model_type': 'classification',
        'num_epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
        'task_trainer': NLVR2Trainer,
        'random_baseline_score': 50.0,
        'low_shot_config': {'task_trainer': LowShotNLVR2Trainer,
                            'type': 'n-shot-per-class',
                            'num_shots_per_class': 2048,
                            'eval_epochs': [6, 8, 10]
                            }
}

snli_ve_config = {
        'task_name': 'SNLI-VE',
        'data_dir': 'snli-ve/',
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
        'task_trainer': SNLIVETrainer,
        'random_baseline_score': 33.33,
        'low_shot_config': {'task_trainer': LowShotSNLIVETrainer,
                            'type': 'n-shot-per-class',
                            'num_shots_per_class': 2048,
                            'eval_epochs': [2, 4, 5]
                            }
}

vcr_config = {
        'task_name': 'VCR',
        'data_dir': 'vcr/',
        'splits': ['train', 'dev', 'test'],
        'num_labels': 4,
        'num_images': 1,
        'model_type': 'multi-choice',
        'task_type': 'answer',
        'num_choices': 4,
        'num_epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
        'task_trainer': VCRTrainer,
        'random_baseline_score': 25.0,
        'low_shot_config': {'task_trainer': LowShotVCRTrainer,
                            'type': 'percentage',
                            'percentage':0.05,
                            'eval_epochs': [2, 4, 6, 8, 10]
                            }
}

imdb_config = {
        'task_name': 'imdb',
        'data_dir': None,
        'cache_dir': '/data/datasets/MCL/cached_datasets',
        'max_len': 160,
        'num_labels': 2,
        'model_type': 'classification',
        'num_epochs': 10,
        'lr': 4e-5,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
}

sst2_config = {
        'task_name': 'sst2',
        'data_dir': None,
        'cache_dir': '/data/datasets/MCL/cached_datasets',
        'max_len': 40,
        'num_labels': 2,
        'model_type': 'classification',
        'num_epochs': 10,
        'lr': 4e-5,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
}


hellaswag_config = {
        'task_name': 'hellaswag',
        'data_dir': '/data/datasets/MCL/hellaswag',
        'max_len': 120,
        'num_labels': 4,
        'model_type': 'classification',
        'num_epochs': 10,
        'lr': 4e-5,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
}


commonsenseqa_config = {
        'task_name': 'commonsenseqa',
        'data_dir': '/data/datasets/MCL/commonsenseqa',
        'max_len': 80,
        'num_labels': 5,
        'model_type': 'classification',
        'num_epochs': 10,
        'lr': 4e-5,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
}


piqa_config = {
        'task_name': 'piqa',
        'data_dir': '/data/datasets/MCL/piqa',
        'max_len': 80,
        'num_labels': 2,
        'model_type': 'classification',
        'num_epochs': 10,
        'lr': 4e-5,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
}


imagenet_config = {
        'task_name': 'imagenet',
        'data_dir': '/data/datasets/MCL/ILSVRC2012',
        'num_labels': 1000,
        'model_type': 'classification',
        'num_epochs': 8,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
}


places365_config = {
        'task_name': 'places365',
        'data_dir': '/data/datasets/MCL/Places365',
        'num_labels': 365,
        'model_type': 'classification',
        'num_epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
}


inat2019_config = {
        'task_name': 'inat2019',
        'data_dir': '/data/datasets/MCL/iNat2019',
        'num_labels': 1010,
        'model_type': 'classification',
        'num_epochs': 8,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
}

coco_cls_config = {
        'task_name': 'coco',
        'data_dir': '/data/datasets/MCL/ms-coco',
        'num_labels': 80,
        'model_type': 'classification',
        'num_epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1,
}

task_configs = {
    'ms-coco': mscoco_config,
    'flickr30k': flickr_config,
    'vqa': vqa_config,
    'nlvr2': nlvr_config,
    'snli-ve': snli_ve_config,
    'vcr': vcr_config,
    'imdb': imdb_config,
    'sst2': sst2_config,
    'hellaswag': hellaswag_config,
    'piqa': piqa_config,
    'commonsenseqa': commonsenseqa_config,
    'imagenet': imagenet_config,
    'places365': places365_config,
    'inat2019': inat2019_config,
    'coco-cls': coco_cls_config,
}
