from .vilt import load_vilt_encoder, ViltContinualLearner
from .viltbert import load_viltbert_encoder, ViltBertContinualLearner

load_encoder_map = {
    'vilt': load_vilt_encoder,
    'viltbert': load_viltbert_encoder
}

continual_learner_map = {
    'vilt': ViltContinualLearner,
    'viltbert': ViltBertContinualLearner
}