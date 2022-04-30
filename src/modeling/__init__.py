from .vilt_modeling import load_vilt_encoder, ViltContinualLearner

load_encoder_map = {
    'vilt': load_vilt_encoder
}

continual_learner_map = {
    'vilt': ViltContinualLearner
}