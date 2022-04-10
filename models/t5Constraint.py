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
import inspect

import torch
from torch import nn
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from transformers import T5PreTrainedModel, T5ForConditionalGeneration, T5Tokenizer, T5Config, T5Model
from transformers.generation_utils import top_k_top_p_filtering

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel, ModelOutput, find_pruneable_heads_and_indices, prune_linear_layer
from torch.utils.checkpoint import checkpoint

from .generation_beam_constraints import Constraint, DisjunctiveConstraint, PhrasalConstraint
from .generation_beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer

from .generation_stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from .generation_logits_process import (
    LogitsProcessorList,
    EncoderNoRepeatNGramLogitsProcessor,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)

from .pytorch_utils import torch_int_div

model_name = 't5-base'


class GreedySearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using greedy search. Hidden states and attention
    weights of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the
    encoder_hidden_states attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)


    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. :obj:`(max_length-1,)`-shaped tuple of :obj:`torch.FloatTensor` with each tensor
            of shape :obj:`(batch_size, config.vocab_size)`).
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer of the decoder) of shape :obj:`(batch_size,
            num_heads, sequence_length, sequence_length)`.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        decoder_attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_heads, generated_length, sequence_length)`.
        cross_attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class GreedySearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. :obj:`(max_length-input_ids.shape[-1],)`-shaped tuple of :obj:`torch.FloatTensor`
            with each tensor of shape :obj:`(batch_size, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class SampleDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using sampling.


    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. :obj:`(max_length-input_ids.shape[-1],)`-shaped tuple of :obj:`torch.FloatTensor`
            with each tensor of shape :obj:`(batch_size*num_return_sequences, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(num_return_sequences*batch_size, num_heads, generated_length,
            sequence_length)`.
        hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(num_return_sequences*batch_size, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class SampleDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using sampling.


    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. :obj:`(max_length-input_ids.shape[-1],)`-shaped tuple of :obj:`torch.FloatTensor`
            with each tensor of shape :obj:`(batch_size*num_return_sequences, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(num_return_sequences*batch_size, num_heads, generated_length,
            sequence_length)`.
        hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(num_return_sequences*batch_size, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class SampleEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using sampling. Hidden states and attention weights of
    the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)


    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. :obj:`(max_length-1,)`-shaped tuple of :obj:`torch.FloatTensor` with each tensor
            of shape :obj:`(batch_size*num_return_sequences, config.vocab_size)`).
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer of the decoder) of shape
            :obj:`(batch_size*num_return_sequences, num_heads, sequence_length, sequence_length)`.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_return_sequences, num_heads, generated_length,
            sequence_length)`.
        cross_attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_return_sequences, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class BeamSearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using beam search. Hidden states and attention weights
    of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)

    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        sequences_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_return_sequences)`, `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Final beam scores of the generated ``sequences``.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . :obj:`(max_length-1,)`-shaped tuple of :obj:`torch.FloatTensor` with each tensor of shape
            :obj:`(batch_size*num_beams, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer of the decoder) of shape :obj:`(batch_size,
            num_heads, sequence_length, sequence_length)`.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams*num_return_sequences, num_heads,
            generated_length, sequence_length)`.
        cross_attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams*num_return_sequences, generated_length,
            hidden_size)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class BeamSearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using beam search.

    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        sequences_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_return_sequences)`, `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Final beam scores of the generated ``sequences``.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . :obj:`(max_length-input_ids.shape[-1],)`-shaped tuple of :obj:`torch.FloatTensor` with each tensor of
            shape :obj:`(batch_size*num_beams*num_return_sequences, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams, num_heads, generated_length,
            sequence_length)`.
        hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams*num_return_sequences, generated_length,
            hidden_size)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class BeamSampleDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using beam sample.

    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        sequences_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_return_sequence)`, `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Final beam scores of the generated ``sequences``.
        scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . :obj:`(max_length-input_ids.shape[-1],)`-shaped tuple of :obj:`torch.FloatTensor` with each tensor of
            shape :obj:`(batch_size*num_beams*num_return_sequences, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams, num_heads, generated_length,
            sequence_length)`.
        hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size*num_beams, generated_length, hidden_size)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput,
                         BeamSearchDecoderOnlyOutput]

# T5PreTrainedModel  PreTrainedModel


class T5ConstrainedGen(T5ForConditionalGeneration):

    def __init__(self, config: T5Config):
        super().__init__(config)

        self.input_ids = None

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
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> # training
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> # inference
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
            >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            >>> # studies have shown that owning a dog is good for you.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                # warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

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
        # print(hidden_states)  hidden_states 固定不变的

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

        # print(decoder_input_ids)  # 上一步的解码输出 第一个为 0

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

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        # if input_ids is None:
        #     input_ids = self.input_ids

        # input_embeds = self.encoder.embed_tokens(input_ids)

        # pointer_logits = torch.einsum('ijk,ilk->ijl', sequence_output, input_embeds)
        # print(pointer_logits)
        # lm_logits1 = self.convert_pointer_logits_to_lm_logits(pointer_logits, input_ids)
        # print(lm_logits1)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

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

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        # 1. retrieve all kwargs that are non-None or non-model input related.
        # some encoder-decoder models have different names for model and encoder
        if (
            self.config.is_encoder_decoder
            and hasattr(self, "encoder")
            and self.encoder.main_input_name != self.main_input_name
        ):
            input_name = self.encoder.main_input_name
        else:
            input_name = self.main_input_name

        model_kwargs = {k: v for k, v in model_kwargs.items(
        ) if v is not None or k != input_name}

        # 2. check whether model_input_name is passed as kwarg
        # if yes and `inputs` is None use kwarg inputs
        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs}` were passed alongside "
                f"{input_name} which is not allowed."
                f"Make sure to either pass {inputs} or {input_name}=..."
            )
        elif inputs_kwarg is not None:
            inputs = inputs_kwarg

        # 3. models with `input_ids` can also make use of `inputs_embeds`
        if self._can_retrieve_inputs_from_name(inputs, "inputs_embeds", model_kwargs):
            inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

        # 4. Only encoder-decoder models can have non `input_ids` input format
        if not self.config.is_encoder_decoder and input_name != "input_ids":
            raise ValueError(
                f"If {input_name} is passed as model-specific keyword "
                "input then model has to be an encoder-decoder and not a "
                f"{self.__class__.__name__}."
            )

        # 5. if `inputs` is still None, try to create `input_ids` from BOS token
        if inputs is None:
            inputs = self._prepare_input_ids_for_generation(
                bos_token_id, model_kwargs.get("encoder_outputs"))

        return inputs, input_name, model_kwargs

    def _get_logits_processor(
        self,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        encoder_no_repeat_ngram_size: int,
        input_ids_seq_length: int,
        encoder_input_ids: torch.LongTensor,
        bad_words_ids: List[List[int]],
        min_length: int,
        max_length: int,
        eos_token_id: int,
        forced_bos_token_id: int,
        forced_eos_token_id: int,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float,
        remove_invalid_values: bool,
        exponential_decay_length_penalty: Tuple,
        logits_processor: Optional[LogitsProcessorList],
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        processors = LogitsProcessorList()

        # init warp parameters
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        encoder_no_repeat_ngram_size = (
            encoder_no_repeat_ngram_size
            if encoder_no_repeat_ngram_size is not None
            else self.config.encoder_no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        min_length = min_length if min_length is not None else self.config.min_length
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        diversity_penalty = diversity_penalty if diversity_penalty is not None else self.config.diversity_penalty
        forced_bos_token_id = (
            forced_bos_token_id if forced_bos_token_id is not None else self.config.forced_bos_token_id
        )
        forced_eos_token_id = (
            forced_eos_token_id if forced_eos_token_id is not None else self.config.forced_eos_token_id
        )
        remove_invalid_values = (
            remove_invalid_values if remove_invalid_values is not None else self.config.remove_invalid_values
        )
        exponential_decay_length_penalty = (
            exponential_decay_length_penalty
            if exponential_decay_length_penalty is not None
            # else self.config.exponential_decay_length_penalty
            else None
        )
        # instantiate processors list

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if diversity_penalty is not None and diversity_penalty > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
                )
            )
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(
                penalty=repetition_penalty))
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            processors.append(
                NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if encoder_no_repeat_ngram_size is not None and encoder_no_repeat_ngram_size > 0:
            if self.config.is_encoder_decoder:
                processors.append(EncoderNoRepeatNGramLogitsProcessor(
                    encoder_no_repeat_ngram_size, encoder_input_ids))
            else:
                raise ValueError(
                    "It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture"
                )
        if bad_words_ids is not None:
            processors.append(NoBadWordsLogitsProcessor(
                bad_words_ids, eos_token_id))
        if min_length is not None and eos_token_id is not None and min_length > 0:
            processors.append(MinLengthLogitsProcessor(
                min_length, eos_token_id))
        if prefix_allowed_tokens_fn is not None:
            processors.append(PrefixConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn, num_beams // num_beam_groups))
        if forced_bos_token_id is not None:
            processors.append(
                ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        if forced_eos_token_id is not None:
            processors.append(ForcedEOSTokenLogitsProcessor(
                max_length, forced_eos_token_id))
        if remove_invalid_values is True:
            processors.append(InfNanRemoveLogitsProcessor())
        if exponential_decay_length_penalty is not None:
            processors.append(
                ExponentialDecayLengthPenalty(
                    exponential_decay_length_penalty, eos_token_id, input_ids_seq_length)
            )
        processors = self._merge_criteria_processor_list(
            processors, logits_processor)
        return processors

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        force_words_ids: Optional[Union[Iterable[int],
                                        Iterable[Iterable[int]]]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[
            int, torch.Tensor], List[int]]] = None,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(
        ),
        stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(
        ),
        constraints: Optional[List[Constraint]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        exponential_decay_length_penalty: Optional[Tuple[Union[int, float]]] = None,
        **model_kwargs,
    ) -> torch.LongTensor:

        # [[ 784,    3,  908, 1820, 3474, 2663, 1465, 7163, 2841, ]]
        # constraint_str = "[ ] | opinion aspect positive neutral negative"
        # constraint_token_ids = tokenizer.encode(constraint_str)[:-1]  # slice to remove eos token

        # pre_input_ids = model_kwargs["input_ids"]
        # pre_input_ids = [set(x for x in ids if x != 0)
        #                         for ids in pre_input_ids.tolist()]
        # pre_input_ids = [list([x] for x in ids )
        #                         for ids in pre_input_ids]
        # force_words_ids = pre_input_ids

        # num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beams = 3

        # 1. Set generation parameters if not already defined
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        if eos_token_id is None and hasattr(self.config, "decoder"):
            eos_token_id = self.config.decoder.eos_token_id

        if pad_token_id is None and eos_token_id is not None:
            # special case if pad_token_id is not defined
            # logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # 2. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["use_cache"] = use_cache

        accepts_attention_mask = "attention_mask" in set(
            inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, pad_token_id, eos_token_id
            )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_token_id,
                model_kwargs=model_kwargs,
            )
        else:
            # if decoder-only then inputs_tensor has to be `input_ids`
            input_ids = inputs_tensor

        input_ids_seq_length = input_ids.shape[-1]

        # 5. Prepare `max_length` depending on other stopping criteria
        # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
        if max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + input_ids_seq_length
        elif max_length is not None and max_new_tokens is not None:
            # Both are set, this is odd, raise a warning
            warnings.warn(
                "Both `max_length` and `max_new_tokens` have been set "
                f"but they serve the same purpose. `max_length` {max_length} "
                f"will take priority over `max_new_tokens` {max_new_tokens}.",
                UserWarning,
            )
        # default to config if still None
        max_length = max_length if max_length is not None else self.config.max_length

        if input_ids_seq_length >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            # logger.warning(
            #     f"Input length of {input_ids_string} is {input_ids_seq_length}, but ``max_length`` is set to {max_length}. "
            #     "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            # )

        # 6. determine generation mode
        is_constraint_gen_mode = constraints is not None or force_words_ids is not None
        # is_greedy_gen_mode = (
        #     (num_beams == 1) and (num_beam_groups == 1) and do_sample is False and not is_constraint_gen_mode
        # )
        # is_sample_gen_mode = (
        #     (num_beams == 1) and (num_beam_groups == 1) and do_sample is True and not is_constraint_gen_mode
        # )
        is_beam_gen_mode = (
            (num_beams > 1) and (num_beam_groups ==
                                 1) and do_sample is False and not is_constraint_gen_mode
        )
        # is_beam_sample_gen_mode = (
        #     (num_beams > 1) and (num_beam_groups == 1) and do_sample is True and not is_constraint_gen_mode
        # )
        # is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1) and not is_constraint_gen_mode

        # if num_beam_groups > num_beams:
        #     raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        # if is_group_beam_gen_mode and do_sample is True:
        #     raise ValueError(
        #         "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
        #     )

        # 7. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            exponential_decay_length_penalty=exponential_decay_length_penalty,
            logits_processor=logits_processor,
        )

        # 8. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
        )
        if is_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now.")

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_constraint_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError(
                    "`max_length` needs to be a stopping_criteria for now.")

            if num_beams <= 1:
                raise ValueError(
                    "`num_beams` needs to be greater than 1 for constrained genertation.")

            if do_sample:
                raise ValueError(
                    "`do_sample` needs to be false for constrained generation.")

            if num_beam_groups is not None and num_beam_groups > 1:
                raise ValueError(
                    "`num_beam_groups` not supported yet for constrained generation.")

            final_constraints = []
            if constraints is not None:
                final_constraints = constraints

            if force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]`"
                        f"of positive integers, but is {force_words_ids}."
                    )

                if not isinstance(force_words_ids, list) or len(force_words_ids) == 0:
                    typeerror()

                for word_ids in force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0)
                                for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 10. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:

            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation_utilsBeamSearchDecoderOnlyOutput`], [`~generation_utils.BeamSearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation_utils.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.


        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt bist du?']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn(
                "You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (
                return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (
            return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get(
                "attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get(
                    "hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(
                input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + \
                beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (
                            outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(
                batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat(
                [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(
                    model_kwargs["past"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple(
                    (beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            else:
                num_return_sequences = beam_scorer.num_beam_hyps_to_keep
                # return only as many indices as sequences
                beam_indices = tuple(
                    (beam_indices[i * num_beams: i * num_beams +
                     num_return_sequences] for i in range(batch_size))
                )
                beam_indices = sum(beam_indices, ())

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=beam_indices,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=beam_indices,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]

    def constrain_search(
        self,
        input_ids,
        beam_scorer=None,
        logits_processor=None,
        stopping_criteria=None,
        # logits_warper=None,
        max_length=None,
        pad_token_id=None,
        eos_token_id=None,
        output_attentions=None,
        output_hidden_states=None,
        output_scores=None,
        return_dict_in_generate=None,
        synced_gpus=None,
        **model_kwargs,
    ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length)
        # logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (
            return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get(
                "attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get(
                    "hidden_states") if output_hidden_states else None
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        # keep track of which sequences are already finished
        # unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        # cur_len = input_ids.shape[-1]

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # TODO! 应该从这里进行修改 不是取最大值，而是取与输入文本 最相似的值 或者 是预定义好的 [ ] | opinion aspect positive  negative neutral

            # 求output和input的相似性？
            # 如何处理 [] 以及 情感词

            # pre-process distribution
            # next_token_scores = logits_processor(input_ids, next_token_logits)
            # print(next_token_scores.shape)

            # 没啥变化 直接返回了 scores ?
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len)

            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)

            # print(next_token_scores)

            # 调整分数
            next_token_scores = next_token_scores + \
                beam_scores[:, None].expand_as(next_token_scores)

            # print(next_token_scores)

            # sample 性能下降
            # next_token_scores = logits_warper(input_ids, next_token_scores)
            # # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (
                            outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )
            # probs = nn.functional.softmax(next_token_scores, dim=-1)
            # next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # argmax
            # next_tokens = torch.argmax(next_token_scores, dim=-1)
            # print(next_tokens)

          # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(
                batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = (next_tokens / vocab_size).long()
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat(
                [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(
                    model_kwargs["past"], beam_idx)

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]

            # TODO! 修改到这里结束

            # # finished sentences should have their next token be a padding token
            # if eos_token_id is not None:
            #     if pad_token_id is None:
            #         raise ValueError(
            #             "If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            #     next_tokens = next_tokens * unfinished_sequences + \
            #         pad_token_id * (1 - unfinished_sequences)

            # # update generated ids, model inputs, and length for next step
            # input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            # # print(input_ids)

            # model_kwargs = self._update_model_kwargs_for_generation(
            #     outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            # )
            # cur_len = cur_len + 1

            # # if eos_token was found in one sentence, set sentence to finished
            # if eos_token_id is not None:
            #     unfinished_sequences = unfinished_sequences.mul(
            #         (next_tokens != eos_token_id).long())

            # # stop when each sentence is finished, or if we exceed the maximum length
            # if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            #     if not synced_gpus:
            #         break
            #     else:
            #         this_peer_finished = True

        # if return_dict_in_generate:
        #     if self.config.is_encoder_decoder:
        #         return GreedySearchEncoderDecoderOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             encoder_attentions=encoder_attentions,
        #             encoder_hidden_states=encoder_hidden_states,
        #             decoder_attentions=decoder_attentions,
        #             cross_attentions=cross_attentions,
        #             decoder_hidden_states=decoder_hidden_states,
        #         )
        #     else:
        #         return GreedySearchDecoderOnlyOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             attentions=decoder_attentions,
        #             hidden_states=decoder_hidden_states,
        #         )
        # else:
        #     return input_ids


def constrained_beam_search(
    self,
    input_ids: torch.LongTensor,
    constrained_beam_scorer: ConstrainedBeamSearchScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: Optional[bool] = None,
    **model_kwargs,
) -> Union[BeamSearchOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **constrained beam search
    decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        constrained_beam_scorer (`ConstrainedBeamSearchScorer`):
            A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation, while satisfying a list of positive constraints. For more information, the
            documentation of [`ConstrainedBeamSearchScorer`] should be read.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        logits_warper (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
            to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`generation_utilsBeamSearchDecoderOnlyOutput`], [`~generation_utils.BeamSearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation_utils.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation_utils.BeamSearchEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.


    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForSeq2SeqLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     ConstrainedBeamSearchScorer,
    ...     PhrasalConstraint,
    ... )
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

    >>> encoder_input_str = "translate English to German: How old are you?"
    >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


    >>> # lets run beam search using 3 beams
    >>> num_beams = 3
    >>> # define decoder start token ids
    >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
    >>> input_ids = input_ids * model.config.decoder_start_token_id

    >>> # add encoder_outputs to model keyword arguments
    >>> model_kwargs = {
    ...     "encoder_outputs": model.get_encoder()(
    ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
    ...     )
    ... }

    >>> constraint_str = "Sie"
    >>> constraint_token_ids = tokenizer.encode(constraint_str)[:-1]  # slice to remove eos token
    >>> constraints = [PhrasalConstraint(token_ids=constraint_token_ids)]


    >>> # instantiate beam scorer
    >>> beam_scorer = ConstrainedBeamSearchScorer(
    ...     batch_size=1, num_beams=num_beams, device=model.device, constraints=constraints
    ... )

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
    ...     ]
    ... )

    >>> outputs = model.constrained_beam_search(
    ...     input_ids, beam_scorer, constraints=constraints, logits_processor=logits_processor, **model_kwargs
    ... )

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Wie alt sind Sie?']
    ```"""
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(
            stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn(
            "You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    output_scores = output_scores if output_scores is not None else self.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (
        return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get(
            "attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get(
                "hidden_states") if output_hidden_states else None
        )

    batch_size = len(constrained_beam_scorer._beam_hyps)
    num_beams = constrained_beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    beam_scores = torch.zeros((batch_size, num_beams),
                              dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only
    while True:

        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(
                0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(
            input_ids, **model_kwargs)

        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
        # cannot be generated both before and after the `nn.functional.log_softmax` operation.
        next_token_logits = self.adjust_logits_during_generation(
            next_token_logits, cur_len=cur_len)
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(
            input_ids, next_token_scores)

        scores_for_all_vocab = next_token_scores_processed.clone()

        next_token_scores = next_token_scores_processed + \
            beam_scores[:, None].expand_as(next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (
                        outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(
            batch_size, num_beams * vocab_size)

        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )

        next_indices = (next_tokens / vocab_size).long()
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = constrained_beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            scores_for_all_vocab,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat(
            [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if model_kwargs["past"] is not None:
            model_kwargs["past"] = self._reorder_cache(
                model_kwargs["past"], beam_idx)

        # increase cur_len
        cur_len = cur_len + 1

        if constrained_beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    sequence_outputs = constrained_beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
    )

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None
        if self.config.is_encoder_decoder:
            return BeamSearchEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return BeamSearchDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return sequence_outputs["sequences"]
