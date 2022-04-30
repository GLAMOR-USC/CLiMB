from modeling.vilt_modeling import *

vilt_config = {
    'encoder_dim': 768,
    'visual_mode': 'pil-image',
    'encoder_class': ViltEncoderWrapper,
    'batch2inputs_converter': convert_batch_to_model_input_dict
}
vilt_lang_seq_config = {
    'encoder_dim': 768,
    'visual_mode': 'pil-image',
    'encoder_class': ViltEncoderWrapper,
    'classifier_class': ViltForSequenceClassification,
    'batch2inputs_converter': convert_seq_batch_to_model_input_dict 
}
vilt_lang_mc_config = {
    'encoder_dim': 768,
    'visual_mode': 'pil-image',
    'encoder_class': ViltEncoderWrapper,
    'classifier_class': ViltForMultipleChoice,
    'batch2inputs_converter': convert_mc_batch_to_model_input_dict
}

model_configs = {
    'vilt': vilt_config,
    'vilt-l-seq': vilt_lang_seq_config,
    'vilt-l-mc': vilt_lang_mc_config
}
