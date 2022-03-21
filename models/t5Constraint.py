# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5 model. """

import copy
import math
from math import ceil
import os
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from transformers import T5PreTrainedModel, T5ForConditionalGeneration, T5Tokenizer, T5Config, T5Model

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from torch.utils.checkpoint import checkpoint

model_name = 't5-base'


# T5PreTrainedModel  PreTrainedModel
class T5ConstrainedGen(T5ForConditionalGeneration):

    def __init__(self, config: T5Config):
        super().__init__(config)

        self.selected_heads = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        # if head_mask is not None and decoder_head_mask is None:
        #     if self.config.num_layers == self.config.num_decoder_layers:
        #         warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
        #         decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(
                    encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(
                    self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device)

        # original outputs
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 1 1 768  batch, seq_len, input_seq_len
        sequence_output = decoder_outputs[0]

        # cross_attention_non_linear = self.decoder.block[-1].layer[1].EncDecAttention.o.weight # (emb_dim, emb_dim)
        # cross_attention_non_linear_sum = cross_attention_non_linear.view(self.config.num_heads, -1).abs().sum(1) # (num_heads)
        # _, selected_heads = torch.topk(cross_attention_non_linear_sum, k=5)
        # self.selected_heads = selected_heads

        # encoder_last_hidden_state = encoder_outputs.last_hidden_state # (batch, seq, hidden)
        # decoder_last_hidden_state = decoder_outputs[0] #(batch, decoding_seq, hidden )

        # compute lm logits based on attention

        # TODO!
        #   print(decoder_outputs.cross_attentions)
        # last_cross_attentions = decoder_outputs.cross_attentions # (batch_size, num_heads, decoding_seq_length, encoding_seq_length).

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)  # torch.Size([1, 7, 32128])

        # TODO!
        # cross_attentions_aggregate = last_cross_attentions[:,self.selected_heads,:,:].mean(dim=1) #(batch, decoding_seq_length, encoding_seq_length)

        # dummy_input_ids = input_ids.unsqueeze(-1).expand(-1, -1, lm_logits.size(1)).transpose(1,2) # (batch, decoding_seq_length, encoding_seq_length)
        # copy_logits = torch.zeros_like(lm_logits) # (batch, decoding_seq_length, emb_dim)
        # copy_logits.scatter_add_(dim=2, index=dummy_input_ids, src=cross_attentions_aggregate)

        # p_gen = torch.bmm(decoder_last_hidden_state, encoder_last_hidden_state.mean(dim=1).unsqueeze(dim=-1)) # (batch, decoding_seq, 1)
        # p_gen = torch.sigmoid(p_gen)

        # lm_logits = F.softmax(lm_logits, dim=-1) * p_gen + copy_logits * (1 - p_gen)#(batch_size, decoding_seq_length, emb_dim)

        # if encoder_outputs==None:
        #     encoder_outputs = outputs[1] # (batch, input_seq_len, hidden_dim)
        #     # BaseModelOutput if return dict

        # print(self.encoder.embed_tokens) # Embedding(32128, 768)
        
        # if inputs_embeds is None:
            # get encoder side embeddings
        # inputs_embeds = self.encoder.embed_tokens(input_ids)
        
        # encoder 一次 decoder N次 ？
        # input_ids 只在第一次输入时存在  tensor([[   37, 32099, 10681,    16, 32098,  2447,     1]])
        # inputs_embeds 一直是 none
        # decoder_input_ids tensor([[    0, 32099,  5295,  1782, 32098,     8, 32097]])
        
        # pointer_logits = torch.einsum(
        #     'ijk,ilk->ijl', sequence_output, inputs_embeds)
        # lm_logits = self.convert_pointer_logits_to_lm_logits(
        #     pointer_logits, input_ids)

        lm_logits = self.lm_head(sequence_output)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def convert_pointer_logits_to_lm_logits(self, pointer_logits, input_ids):
        '''
        pointer_logits: (batch, seq_len, input_seq_len)
        input_ids: (batch, input_seq_len)
        lm_logits: (batch, seq_len, vocab_size)
        '''
        batch_size = pointer_logits.size(0)
        seq_len = pointer_logits.size(1)
        # input_seq_len = input_ids.size(1)
        lm_logits = torch.full((batch_size, seq_len, self.config.vocab_size),
                               fill_value=-1000, dtype=pointer_logits.dtype).to(pointer_logits.device)

        #  scatter may be technically incorrect for duplicate indexes, but not using it gets slow
        index = input_ids.unsqueeze(dim=1).expand_as(pointer_logits)
        lm_logits.scatter_(dim=2, index=index, src=pointer_logits)

        return lm_logits
