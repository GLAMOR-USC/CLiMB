from modeling.vilt_modeling import ViltEncoder, ViltForImageTextClassification

model_configs = {
    'vilt': vilt_config
}

vilt_config = {
    'encoder_dim': 768,
    'visual_mode': 'pil-image',
    'processor': ViltProcessor,
    'vilt': ViltModel,
    'encoder_class': ViltEncoder,
    'classifier_class': ViltForImageTextClassification
}