"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for VQA model
"""
from collections import defaultdict

from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .layer import GELU
from .model import UniterPreTrainedModel, UniterModel
from KD_loss import distillation_loss, patience_loss



class UniterForVisualQuestionAnswering(UniterPreTrainedModel):
    """ Finetune UNITER for VQA
    """
    def __init__(self, config, img_dim, num_answer):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.vqa_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            GELU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, num_answer)
        )
        self.apply(self.init_weights)

    def forward(self, batch, adv_training=False, adv_modality=None,
                adv_delta_txt=None, adv_delta_img=None, compute_loss=True, is_val = False, return_attention=False, return_embedding=False):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
    
        # all_encoder_layers =  self.uniter(input_ids, position_ids,
        #                               img_feat, img_pos_feat,
        #                               attn_masks, adv_training,
        #                               adv_modality, adv_delta_txt,
        #                               adv_delta_img, gather_index,
        #                               output_all_encoded_layers=True)

        if return_attention and return_embedding:

            img_txt_all_encoder_layers, img_emb, txt_emb, embedding_output, attention_list = self.uniter(input_ids, position_ids,
                                                     img_feat, img_pos_feat,
                                                     attn_masks, adv_training,
                                                     adv_modality, adv_delta_txt,
                                                     adv_delta_img, gather_index,
                                                     output_all_encoded_layers=True, return_attention=return_attention, return_embedding=return_embedding)
        else:
            img_txt_all_encoder_layers = self.uniter(input_ids, position_ids,
                                                     img_feat, img_pos_feat,
                                                     attn_masks, adv_training,
                                                     adv_modality, adv_delta_txt,
                                                     adv_delta_img, gather_index,
                                                     output_all_encoded_layers=True)

        img_txt_sequence_output = img_txt_all_encoder_layers[-1]
        img_txt_pooled_output = self.uniter.pooler(img_txt_sequence_output)
        img_txt_answer_scores = self.vqa_output(img_txt_pooled_output)

        if not is_val:
            img_all_encoder_layers = self.uniter(None, position_ids,
                                                 img_feat, img_pos_feat,
                                                 attn_masks, adv_training,
                                                 adv_modality, adv_delta_txt,
                                                 adv_delta_img, gather_index,
                                                 output_all_encoded_layers=True)
            txt_all_encoder_layers = self.uniter(input_ids, position_ids,
                                                 None, img_pos_feat,
                                                 attn_masks, adv_training,
                                                 adv_modality, adv_delta_txt,
                                                 adv_delta_img, gather_index,
                                                 output_all_encoded_layers=True)


            img_sequence_output = img_all_encoder_layers[-1]
            img_pooled_output = self.uniter.pooler(img_sequence_output)
            img_answer_scores = self.vqa_output(img_pooled_output)

            txt_sequence_output = txt_all_encoder_layers[-1]
            txt_pooled_output = self.uniter.pooler(txt_sequence_output)
            txt_answer_scores = self.vqa_output(txt_pooled_output)


        if compute_loss:
            targets = batch['targets']
            vqa_loss = F.binary_cross_entropy_with_logits(
                img_txt_answer_scores, targets, reduction='none')
            return vqa_loss
        else:
            if is_val:
                if return_attention and return_embedding:
                    return img_txt_answer_scores, img_txt_all_encoder_layers, img_emb, txt_emb, embedding_output, attention_list
                else:
                    return img_txt_answer_scores, img_txt_all_encoder_layers
            else:
                return img_all_encoder_layers, img_answer_scores, txt_all_encoder_layers, txt_answer_scores, img_txt_all_encoder_layers, img_txt_answer_scores
