from modeling.vilt_modeling import *

vilt_config = {
    'encoder_dim': 768,
    'visual_mode': 'pil-image',
    'encoder_class': ViltEncoder,
    'classifier_class': ViltForImageTextClassification
}

model_configs = {
    'vilt': vilt_config
}
