from modeling.vilt_modeling import *

vilt_config = {
    'encoder_dim': 768,
    'visual_mode': 'pil-image',
    'encoder_class': ViltEncoderWrapper,
    'batch2inputs_converter': convert_batch_to_model_input_dict
}

model_configs = {
    'vilt': vilt_config
}
