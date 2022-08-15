from .vilt import load_vilt_encoder, create_vilt_continual_learner_model
from .viltbert import load_viltbert_encoder, create_viltbert_continual_learner_model

load_encoder_map = {
    'vilt': load_vilt_encoder,
    'viltbert': load_viltbert_encoder
}

create_continual_learner_map = {
    'vilt': create_vilt_continual_learner_model,
    'viltbert': create_viltbert_continual_learner_model,
}