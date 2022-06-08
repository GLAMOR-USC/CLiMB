from modeling.vilt import *
from modeling.viltbert import *

ALLOWED_CL_ENCODERS = ['vilt', 'viltbert']

vilt_config = {
    'encoder_dim': 768,
    'visual_mode': 'pil-image',
    'encoder_class': ViltEncoderWrapper,
    'batch2inputs_converter': convert_batch_to_vilt_input_dict,
    'encoder_name': 'ViLT'
}

vilt_lang_seq_config = {
    'encoder_dim': 768,
    'visual_mode': 'pil-image',
    'encoder_class': ViltEncoderWrapper,
    'classifier_class': ViltForSequenceClassification,
    'batch2inputs_converter': convert_seq_batch_to_vilt_input_dict 
}
vilt_lang_mc_config = {
    'encoder_dim': 768,
    'visual_mode': 'pil-image',
    'encoder_class': ViltEncoderWrapper,
    'classifier_class': ViltForMultipleChoice,
    'batch2inputs_converter': convert_mc_batch_to_vilt_input_dict
}
vilt_vision_cls_config = {
    'encoder_dim': 768,
    'visual_mode': 'pil-image',
    'encoder_class': ViltEncoderWrapper,
    'classifier_class': ViltForImageClassification,
    'batch2inputs_converter': convert_batch_to_vilt_input_dict
}

viltbert_config = {
    'encoder_dim': 768,
    'visual_mode': 'pil-image',
    'encoder_class': ViltBertEncoderWrapper,
    'batch2inputs_converter': convert_batch_to_viltbert_input_dict,
    'encoder_name': 'ViLT-BERT'
}
viltbert_lang_seq_config = {
    'encoder_dim': 768,
    'visual_mode': 'pil-image',
    'encoder_class': ViltEncoderWrapper,
    'classifier_class': ViltBertForSequenceClassification,
    'batch2inputs_converter': convert_seq_batch_to_vilt_input_dict 
}
viltbert_lang_mc_config = {
    'encoder_dim': 768,
    'visual_mode': 'pil-image',
    'encoder_class': ViltEncoderWrapper,
    'classifier_class': ViltBertForMultipleChoice,
    'batch2inputs_converter': convert_mc_batch_to_vilt_input_dict
}

model_configs = {
    'vilt': vilt_config,
    'vilt-v-cls': vilt_vision_cls_config,
    'vilt-l-seq': vilt_lang_seq_config,
    'vilt-l-mc': vilt_lang_mc_config,
    'viltbert': viltbert_config,
    'viltbert-l-seq': viltbert_lang_seq_config, 
    'viltbert-l-mc': viltbert_lang_mc_config
}
