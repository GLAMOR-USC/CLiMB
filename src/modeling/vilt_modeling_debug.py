import os
import sys
import logging
import itertools
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, BertTokenizer, BertModel
from transformers import ViltProcessor, ViltForImagesAndTextClassification

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

def debug(processor, encodings, n=4):
    from torchvision.utils import save_image
    def denorm(x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    imgs = denorm(encodings['pixel_values'][: n*2].cpu())
    texts = processor.batch_decode(encodings['input_ids'][:n])
    save_image(imgs, 'debug_img.png', nrow=2, padding=0)
    print(texts)
    pdb.set_trace()



class ViltForImageTextClassification(nn.Module):
    # rewrite class ViltForImagesAndTextClassification
    # https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/vilt/modeling_vilt.py#L1270

    def __init__(self, processor, model, device):

        super().__init__()

        self.processor = processor
        self.model = model
        self.device = device


    def process_inputs(self, images, texts):
        #TODO: trucation somehow triggers "Ignored unknown kwarg option direction"
        encodings = self.processor(images=images, text=texts, 
            padding=True, truncation=True, return_tensors='pt').to(self.device)

        #debug(self.processor, encodings)
        return encodings

    def forward(self, images, texts, num_images=2):
        # flatten n images & pre-process
        flat_images_list = list(itertools.chain(*images))
        encodings = self.process_inputs(flat_images_list, texts)

        input_ids, attention_mask, token_type_ids = \
            encodings['input_ids'], encodings['attention_mask'], encodings['token_type_ids']
        # reshape
        bs = len(input_ids)
        pixel_values = encodings['pixel_values'].view(bs, num_images, *encodings["pixel_values"].shape[-3:])
        pixel_mask = encodings['pixel_mask'].view(bs, num_images, *encodings["pixel_mask"].shape[-2:])

        # https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/vilt/modeling_vilt.py#L1351
        pooler_outputs = []
        for i in range(num_images):
            # forward every image through the model
            encodings = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'pixel_values': pixel_values[:, i, :, :, :],
                'pixel_mask': pixel_mask[:, i, :, :] if pixel_mask is not None else None, #TODO, None in the example code
                'image_token_type_idx': i + 1, #TODO: somehow triggers error when PLM='dandelin/vilt-b32-mlm'
            }
            pooled_out = self.model.vilt(**encodings).pooler_output
            pooler_outputs.append(pooled_out)
        pooled_output = torch.cat(pooler_outputs, dim=-1) # [bs, 1536]

        output_logits = self.model.classifier(pooled_output)
        return pooled_output, output_logits

def load_vilt(pretrained_vilt_name, device):

    logger.info("-"*100)
    logger.info("Loading pretrained ViLT model: {}".format(pretrained_vilt_name))
    vilt_processor = ViltProcessor.from_pretrained(pretrained_vilt_name)
    vilt_model = ViltForImagesAndTextClassification.from_pretrained(pretrained_vilt_name)
    model = ViltForImageTextClassification(vilt_processor, vilt_model, device)
    logger.info("Successfully loaded pretrained ViLT model")
    return model

def convert_batch_to_model_input_dict(batch):

    return {'images': batch['images'],
            'texts': batch['raw_texts']}
