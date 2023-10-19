# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch BART model. """
import warnings

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import logging
from transformers import BartConfig, BartModel, BartPretrainedModel
from typing import Optional, Union, Iterable, Callable, List, Tuple, Dict, Any
from transformers.generation_utils import (BeamSearchDecoderOnlyOutput, BeamSearchEncoderDecoderOutput,)
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.file_utils import ModelOutput
from transformers.generation_logits_process import LogitsProcessorList
from transformers.generation_stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from dataclasses import dataclass
from torch.nn.modules.loss import _Loss
import copy

logger = logging.get_logger(__name__)
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]


_CHECKPOINT_FOR_DOC = "facebook/bart-large"
_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"


BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-large",
    # See all BART models at https://huggingface.co/models?filter=bart
]


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


BART_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BartConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BART_GENERATION_EXAMPLE = r"""
    Summarization example::

        >>> from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

        >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

        >>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

        >>> # Generate Summary
        >>> summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        >>> print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

    Mask filling example::

        >>> from transformers import BartTokenizer, BartForConditionalGeneration
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        >>> TXT = "My friends are <mask> but they eat too many carbs."

        >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
        >>> logits = model(input_ids).logits

        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
"""

BART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are decoder input IDs? <../glossary.html#decoder-input-ids>`__

            Bart uses the :obj:`eos_token_id` as the starting token for :obj:`decoder_input_ids` generation. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).

            For translation and summarization training, :obj:`decoder_input_ids` should be provided. If no
            :obj:`decoder_input_ids` is provided, the model will create this tensor by shifting the :obj:`input_ids` to
            the right for denoising pre-training following the paper.
        decoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should read :func:`modeling_bart._prepare_decoder_inputs` and
            modify to your needs. See diagram 1 in `the paper <https://arxiv.org/abs/1910.13461>`__ for more
            information on the default strategy.
        head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in ``[0,
            1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, `optional`: :obj:`hidden_states`, `optional`:
            :obj:`attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`,
            `optional`) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
            cross-attention of the decoder.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds`
            have to be input (see :obj:`past_key_values`). This is useful if you want more control over how to convert
            :obj:`decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset, :obj:`decoder_inputs_embeds`
            takes the value of :obj:`inputs_embeds`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


class GaussianKLLoss(_Loss):
    def __init__(self):
        super(GaussianKLLoss, self).__init__()

    def forward(self, mu, logvar, mu_prior=None, logvar_prior=None):
        """
        0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        torch.log(logvar_prior.exp().pow(0.5) / logvar.exp().pow(0.5)) + \
                        0.5 * ((mu - mu_prior).pow(2) + logvar.exp()) / logvar_prior.exp() - 0.5
        """
        if mu_prior is None and logvar_prior is None:
            y_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        else:
            KLD_element = torch.log(logvar_prior.exp() / logvar.exp()) + \
                            ((mu - mu_prior).pow(2) + logvar.exp()) / logvar_prior.exp() - 1
            y_kl = 0.5 * torch.sum(KLD_element, dim=-1)

        return torch.mean(y_kl)


class Pack(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            return super(Pack, self).__getattr__(name)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack

@dataclass
class Seq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """
    prediction_loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    reconstruction_loss: Optional[torch.FloatTensor] = None
    reconstruction_logits: torch.FloatTensor = None
    kl_loss: Optional[torch.FloatTensor] = None
    topic_constraint: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values_for_x: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states_for_x: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions_for_x: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    latent_result: Pack = None

@dataclass
class Seq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    last_hidden_state_for_x: torch.FloatTensor = None
    past_key_values_for_x: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states_for_x: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions_for_x: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    causal_latent: torch.FloatTensor = None
    non_causal_latent: torch.FloatTensor = None
    latent_result: Pack = None
    style_doc_latent: torch.FloatTensor = None
    style_summary_latent: torch.FloatTensor = None


@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.",
    BART_START_DOCSTRING,
)
class BartModel_VAE(BartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.aggregate = config.aggregate
        self.training_vae = config.training_vae
        self.fuse_seq_info = config.fuse_seq_info
        self.causal_latent_size = config.causal_latent_size
        self.non_causal_latent_size = config.non_causal_latent_size
        self.style_size = config.style_size
        self.total_latent_size = self.causal_latent_size + self.non_causal_latent_size + self.style_size
        self.u_num_embeddings = config.num_topics
        self.u_size = config.u_size
        self.u_embed = torch.nn.Embedding(self.u_num_embeddings, self.u_size)
        self.q_z_mu = nn.Linear(config.d_model + self.u_size, self.total_latent_size)
        self.q_z_logvar = nn.Linear(config.d_model + self.u_size, self.total_latent_size)

        self.size_transform_fct_for_causal = nn.Linear(self.causal_latent_size, config.d_model)
        self.size_transform_fct_for_non_causal = nn.Linear(self.non_causal_latent_size, config.d_model)
        self.size_transform_fct_for_style = nn.Linear(self.style_size, config.d_model)

        self.decoder_for_x = copy.deepcopy(self.decoder)

        # Initialize weights and apply final processing
        self.init_weights()

    def reparameterize(self, mu, logvar):
        if self.training_vae:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std) # size = input.shape
            return eps.mul(std).add_(mu) # z = mu + std * eps
        else:
            return mu

    def latent_sample_u(self, encoded_seq, u):
        if self.fuse_seq_info:
            if self.aggregate == "cls":
                encoded_seq = encoded_seq[:,1,:].unsqueeze(1)
            else: # average pool
                encoded_seq = encoded_seq.mean(1).unsqueeze(1)

        u = u[:, None, :]
        t = torch.concat([encoded_seq, u], dim=-1)
        z_mu = self.q_z_mu(t)
        z_logvar = self.q_z_logvar(t)
        sample_z = self.reparameterize(z_mu, z_logvar)
        return Pack(sample_z=sample_z, z_mu=z_mu, z_logvar=z_logvar)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder_for_x.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder_for_x(self):
        return self.decoder_for_x

    def get_decoder_for_y(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        target,
        cr,
        subset_id,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_input_ids_for_x=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        past_key_values_for_x=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        latent_result=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if target == "prediction" or target == "both":
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # for restructing x
        if target == "reconstruction" or target == "both":
            if decoder_input_ids_for_x is None:
                decoder_input_ids_for_x = shift_tokens_right(
                    input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # VAE latent variables
        if latent_result is None:
            # encoder_outputs[0]: encoder_outputs.last_hidden_state
            # (batch_size, encoder_sequence_length, hidden_size)
            encoder_seq_hidden_state = encoder_outputs.last_hidden_state if return_dict else encoder_outputs[0] # outputs.encoder_last_hidden_state
            # (batch_size, 1, total_latent_size) if fuse_seq_info
            subset_embedding = self.u_embed(subset_id)
            latent_result = self.latent_sample_u(encoder_seq_hidden_state, subset_embedding)

        # (batch_size, 1, total_latent_size) if fuse_seq_info
        total_latent = latent_result.sample_z

        # separate into three latent variables
        # (batch_size, 1, causal_latent_size)
        causal_latent = total_latent[:,:,:self.causal_latent_size]
        # (batch_size, 1, hidden_size)
        causal_latent = self.size_transform_fct_for_causal(causal_latent)

        non_causal_latent=None
        if target == "reconstruction" or target == "both":
            # (batch_size, 1, non_causal_latent_size)
            non_causal_latent = total_latent[:,:,self.causal_latent_size : self.causal_latent_size + self.non_causal_latent_size]
            # (batch_size, 1, hidden_size)
            non_causal_latent = self.size_transform_fct_for_non_causal(non_causal_latent)

        style_doc_latent = total_latent[:,:,-self.style_size:]
        style_doc_latent = self.size_transform_fct_for_style(style_doc_latent)
        style_summary_latent = style_doc_latent * cr[:,None,None]

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs_for_x = None
        if target == "reconstruction" or target == "both":
            if past_key_values_for_x is None:
                decoder_inputs_embeds_for_x = self.decoder_for_x.embed_tokens(decoder_input_ids_for_x) * self.decoder_for_x.embed_scale
                new_decoder_inputs_embeds_for_x = torch.concat([decoder_inputs_embeds_for_x[:,:1,:],causal_latent+non_causal_latent+style_doc_latent,decoder_inputs_embeds_for_x[:,2:,:]],dim=1)

            # total_latent_for_reconstruction
            decoder_outputs_for_x = self.decoder_for_x(
                input_ids=None if past_key_values_for_x is None else decoder_input_ids_for_x,
                attention_mask=None,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                head_mask=None,
                cross_attn_head_mask=None,
                past_key_values=past_key_values_for_x,
                inputs_embeds=new_decoder_inputs_embeds_for_x if past_key_values_for_x is None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # print("Get decoder_outputs_for_x!")

        decoder_outputs = None
        if target == "prediction" or target == "both":
            new_decoder_inputs_embeds = None
            if past_key_values is None and decoder_input_ids.shape[1]>1: # train
                if decoder_inputs_embeds is None:
                    decoder_inputs_embeds = self.decoder.embed_tokens(decoder_input_ids) * self.decoder.embed_scale
                new_decoder_inputs_embeds = torch.concat([decoder_inputs_embeds[:,:1,:],causal_latent+style_summary_latent,decoder_inputs_embeds[:,2:,:]],dim=1)

            elif decoder_input_ids.shape[1]==1 and decoder_input_ids[0,0].item()==0 and past_key_values[0][0].shape[2]<3: # test, real_start
                if decoder_inputs_embeds is None:
                    decoder_inputs_embeds = self.decoder.embed_tokens(decoder_input_ids) * self.decoder.embed_scale
                new_decoder_inputs_embeds = causal_latent+style_summary_latent

            # causal_latent
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids if new_decoder_inputs_embeds is None else None,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds if new_decoder_inputs_embeds is None else new_decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # print("Get decoder_outputs!")

        if not return_dict:
            # return decoder_outputs + encoder_outputs
            # return tuple: concat of tuple is tuple, "(xxx,)" should be added for tensor
            # each element of the result tuple is the elements of all source tuples
            tuple_output = encoder_outputs  + (causal_latent, non_causal_latent, latent_result, style_doc_latent, style_summary_latent)
            if decoder_outputs_for_x is not None:
                tuple_output = decoder_outputs_for_x + tuple_output
            if decoder_outputs is not None:
                tuple_output = decoder_outputs + tuple_output
            return tuple_output

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state if decoder_outputs is not None else None,
            past_key_values=decoder_outputs.past_key_values if decoder_outputs is not None else None,
            decoder_hidden_states=decoder_outputs.hidden_states if decoder_outputs is not None else None,
            decoder_attentions=decoder_outputs.attentions if decoder_outputs is not None else None,
            cross_attentions=decoder_outputs.cross_attentions if decoder_outputs is not None else None,
            last_hidden_state_for_x=decoder_outputs_for_x.last_hidden_state if decoder_outputs_for_x is not None else None,
            past_key_values_for_x=decoder_outputs_for_x.past_key_values if decoder_outputs_for_x is not None else None,
            decoder_hidden_states_for_x=decoder_outputs_for_x.hidden_states if decoder_outputs_for_x is not None else None,
            cross_attentions_for_x=decoder_outputs_for_x.cross_attentions if decoder_outputs_for_x is not None else None,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            causal_latent=causal_latent,
            non_causal_latent=non_causal_latent,
            latent_result=latent_result,
            style_doc_latent=style_doc_latent,
            style_summary_latent=style_summary_latent,
        )


@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.", BART_START_DOCSTRING
)
class BartForConditionalGeneration_VAE(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel_VAE(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.lm_head_for_x = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.topic_weight = nn.Parameter(torch.rand(config.num_topics,config.d_model))

        # Initialize weights and apply final processing
        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder_for_x(self):
        return self.model.get_decoder_for_x()

    def get_decoder_for_y(self):
        return self.model.get_decoder_for_y()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
        self,
        target,
        cr,
        subset_id,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_input_ids_for_x=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        past_key_values_for_x=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        labels_for_x=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        latent_result=None,
        topic_distribution=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if labels_for_x is not None:
            if decoder_input_ids_for_x is None:
                decoder_input_ids_for_x = shift_tokens_right(
                    labels_for_x, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            target,
            cr,
            subset_id,
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_input_ids_for_x=decoder_input_ids_for_x,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            past_key_values_for_x=past_key_values_for_x,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            latent_result=latent_result,
        )

        causal_latent = outputs.causal_latent
        non_causal_latent = outputs.non_causal_latent
        style_doc_latent = outputs.style_doc_latent
        style_summary_latent = outputs.style_summary_latent

        topic_constraint = None
        if topic_distribution is not None:

            tc_gate = torch.clip(torch.sign(topic_distribution - self.config.tc_th), 1e-7, -1e-7 + 1)
            tc_weight = tc_gate * topic_distribution
            tc_distribution = tc_weight / (torch.sum(tc_weight,dim=1)).unsqueeze(1)
            tnc_gate = 1 - tc_gate
            tnc_weight = tnc_gate * topic_distribution
            tnc_distribution = tnc_weight / (torch.sum(tnc_weight,dim=1)).unsqueeze(1)

            topic_causal_embedding = torch.matmul(tc_distribution, self.topic_weight)
            topic_non_causal_embedding = torch.matmul(tnc_distribution, self.topic_weight)

            euclidean_distance1 = torch.pairwise_distance(causal_latent, topic_causal_embedding)
            euclidean_distance2 = torch.pairwise_distance(non_causal_latent, topic_non_causal_embedding)
            topic_constraint = torch.mean(euclidean_distance1) + torch.mean(euclidean_distance2)

        lm_logits = None
        lm_logits_for_x = None
        if target == "prediction" or target == "both":
            # [bs,seq_len,hidden_size]
            decoder_last_hidden_state = outputs[0]
            # [bs,seq_len,hidden_size]
            z_latent = causal_latent + style_summary_latent
            z_latent = z_latent.repeat([1,decoder_last_hidden_state.shape[1],1])
            decoder_last_hidden_state = decoder_last_hidden_state + z_latent
            lm_logits = self.lm_head(decoder_last_hidden_state) + self.final_logits_bias

        if target == "both":
            decoder_last_hidden_state_for_x = outputs[2] if not return_dict else outputs.last_hidden_state_for_x
            z_latent = causal_latent + non_causal_latent + style_doc_latent
            z_latent = z_latent.repeat([1,decoder_last_hidden_state_for_x.shape[1],1])
            decoder_last_hidden_state_for_x = decoder_last_hidden_state_for_x + z_latent
            lm_logits_for_x = self.lm_head_for_x(decoder_last_hidden_state_for_x) + self.final_logits_bias
        elif target == "reconstruction":
            decoder_last_hidden_state_for_x = outputs[0]
            z_latent = causal_latent + non_causal_latent + style_doc_latent
            z_latent = z_latent.repeat([1,decoder_last_hidden_state_for_x.shape[1],1])
            decoder_last_hidden_state_for_x = decoder_last_hidden_state_for_x + z_latent
            lm_logits_for_x = self.lm_head_for_x(decoder_last_hidden_state_for_x) + self.final_logits_bias


        lm_loss_for_y = None
        if labels is not None:
            loss_fct_y = CrossEntropyLoss()
            lm_loss_for_y = loss_fct_y(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        lm_loss_for_x = None
        if labels_for_x is not None:
            loss_fct_x = CrossEntropyLoss()
            lm_loss_for_x = loss_fct_x(lm_logits_for_x.view(-1, self.config.vocab_size), labels_for_x.view(-1))

        vae_kl_loss = None
        if self.model.training_vae:
            kl_loss_fct = GaussianKLLoss()
            latent_mu = outputs.latent_result.z_mu if return_dict else outputs[-1].z_mu
            latent_logvar = outputs.latent_result.z_logvar if return_dict else outputs[-1].z_logvar
            if self.model.learned_prior:
                latent_mu_prior = outputs.latent_result.z_mu_prior if return_dict else outputs[-1].z_mu_prior
                latent_logvar_prior = outputs.latent_result.z_logvar_prior if return_dict else outputs[-1].z_logvar_prior
                vae_kl_loss = kl_loss_fct(latent_mu, latent_logvar, latent_mu_prior, latent_logvar_prior)
            else:
                vae_kl_loss = kl_loss_fct(latent_mu, latent_logvar)

        if not return_dict:
            output = outputs[-2:]

            if target == "both":
                past_key_values_for_x = outputs[3]
                output = ((past_key_values_for_x,) + output)
            elif target == "reconstruction":
                past_key_values_for_x = outputs[1]
                output = ((past_key_values_for_x,) + output)

            if target == "prediction" or target == "both":
                past_key_values = outputs[1]
                output = ((past_key_values,) + output)

            if vae_kl_loss is not None:
                output = ((vae_kl_loss,) + output)
            if lm_loss_for_x is not None:
                output = ((lm_loss_for_x,) + (lm_logits_for_x,) + output)
            if lm_loss_for_y is not None:
                output = ((lm_loss_for_y,) + (lm_logits,) + output)
            return output

        return Seq2SeqLMOutput(
            prediction_loss=lm_loss_for_y,
            prediction_logits=lm_logits,
            reconstruction_loss=lm_loss_for_x,
            reconstruction_logits=lm_logits_for_x,
            kl_loss=vae_kl_loss,
            topic_constraint=topic_constraint,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            past_key_values_for_x=outputs.past_key_values_for_x,
            decoder_hidden_states_for_x=outputs.decoder_hidden_states_for_x,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            latent_result=outputs.latent_result,
        )


    @torch.no_grad()
    def generate(
        self,
        target,
        cr,
        subset_id,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
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
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        latent_result: Optional[Pack] = None,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        multinomial sampling, beam-search decoding, and beam-search multinomial sampling.

        Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
        attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
        indicated are the default values of those config.

        Most of these parameters are explained in more detail in `this blog post
        <https://huggingface.co/blog/how-to-generate>`__.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it with
                :obj:`bos_token_id` and a batch size of 1.
            max_length (:obj:`int`, `optional`, defaults to :obj:`model.config.max_length`):
                The maximum length of the sequence to be generated.
            max_new_tokens (:obj:`int`, `optional`, defaults to None):
                The maximum numbers of tokens to generate, ignore the current number of tokens. Use either
                :obj:`max_new_tokens` or :obj:`max_length` but not both, they serve the same purpose.
            min_length (:obj:`int`, `optional`, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
                model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
                sequences.
            no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            encoder_no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size that occur in the ``encoder_input_ids`` cannot occur in the
                ``decoder_input_ids``.
            bad_words_ids(:obj:`List[List[int]]`, `optional`):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use :obj:`tokenizer(bad_word,
                add_prefix_space=True).input_ids`.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            max_time(:obj:`float`, `optional`, defaults to None):
                The maximum amount of time you allow the computation to run for in seconds. generation will still
                finish the current pass after allocated time has been passed.
            attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
                tokens that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same
                shape as :obj:`input_ids` that masks the pad token. `What are attention masks?
                <../glossary.html#attention-mask>`__
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            num_beam_groups (:obj:`int`, `optional`, defaults to 1):
                Number of groups to divide :obj:`num_beams` into in order to ensure diversity among different groups of
                beams. `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
            diversity_penalty (:obj:`float`, `optional`, defaults to 0.0):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that :obj:`diversity_penalty` is only effective if ``group beam search`` is
                enabled.
            prefix_allowed_tokens_fn: (:obj:`Callable[[int, torch.Tensor], List[int]]`, `optional`):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID :obj:`batch_id` and
                :obj:`input_ids`. It has to return a list with the allowed tokens for the next generation step
                conditioned on the batch ID :obj:`batch_id` and the previously generated tokens :obj:`inputs_ids`. This
                argument is useful for constrained generation conditioned on the prefix, as described in
                `Autoregressive Entity Retrieval <https://arxiv.org/abs/2010.00904>`__.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            forced_bos_token_id (:obj:`int`, `optional`):
                The id of the token to force as the first generated token after the :obj:`decoder_start_token_id`.
                Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token
                needs to be the target language token.
            forced_eos_token_id (:obj:`int`, `optional`):
                The id of the token to force as the last generated token when :obj:`max_length` is reached.
            remove_invalid_values (:obj:`bool`, `optional`):
                Whether to remove possible `nan` and `inf` outputs of the model to prevent the generation method to
                crash. Note that using ``remove_invalid_values`` can slow down generation.
            synced_gpus (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)

            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If the
                model is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific
                kwargs should be prefixed with `decoder_`.

        Return:
            :class:`~transformers.file_utils.ModelOutput` or :obj:`torch.LongTensor`: A
            :class:`~transformers.file_utils.ModelOutput` (if ``return_dict_in_generate=True`` or when
            ``config.return_dict_in_generate=True``) or a :obj:`torch.FloatTensor`.

                If the model is `not` an encoder-decoder model (``model.config.is_encoder_decoder=False``), the
                possible :class:`~transformers.file_utils.ModelOutput` types are:

                    - :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.SampleDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.BeamSearchDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.BeamSampleDecoderOnlyOutput`

                If the model is an encoder-decoder model (``model.config.is_encoder_decoder=True``), the possible
                :class:`~transformers.file_utils.ModelOutput` types are:

                    - :class:`~transformers.generation_utils.GreedySearchEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.SampleEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.BeamSampleEncoderDecoderOutput`

        Examples::
            >>> from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

            >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
            >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
            >>> document = (
            ... "at least two people were killed in a suspected bomb attack on a passenger bus "
            ... "in the strife-torn southern philippines on monday , the military said."
            ... )
            >>> # encode input context
            >>> input_ids = tokenizer(document, return_tensors="pt").input_ids
            >>> # generate 3 independent sequences using beam search decoding (5 beams)
            >>> # with T5 encoder-decoder model conditioned on short news article.
            >>> outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

        """

        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states

        if input_ids is None and "inputs_embeds" not in model_kwargs:
            # init `input_ids` with bos_token_id
            input_ids = self._prepare_input_ids_for_generation(bos_token_id, model_kwargs.get("encoder_outputs"))

        if model_kwargs.get("attention_mask", None) is None:
            # init `attention_mask` depending on `pad_token_id`
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )

        # special case if pad_token_id is not defined
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        # Storing encoder_input_ids for logits_processor that could use them
        encoder_input_ids = input_ids if self.config.is_encoder_decoder else None

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)

            # set input_ids as decoder_input_ids
            if "decoder_input_ids" in model_kwargs:
                input_ids = model_kwargs.pop("decoder_input_ids")
            else:
                # input encoder_input_ids for bs
                input_ids = self._prepare_decoder_input_ids_for_generation(
                    input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id
                )

            if "encoder_outputs" not in model_kwargs or not isinstance(model_kwargs["encoder_outputs"], ModelOutput):
                raise ValueError("Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`.")

        # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
        if max_length is None and max_new_tokens is not None:
            max_length = (
                max_new_tokens + input_ids.shape[-1]
                if input_ids is not None
                else max_length + model_kwargs["inputs_embeds"].shape[1]
            )
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

        if input_ids.shape[-1] >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}. "
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )

        # determine generation mode
        is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")

        # set model_kwargs
        model_kwargs["use_cache"] = use_cache

        # get distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            encoder_input_ids=encoder_input_ids,
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
        )

        stopping_criteria = self._get_stopping_criteria(max_length=max_length, max_time=max_time)
        
        
        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )


        elif is_beam_gen_mode:
            batch_size = input_ids.shape[0]

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # interleave with `num_beams`

            input_ids, cr, subset_id, latent_result, model_kwargs = self._expand_inputs_for_generation(
                input_ids, cr, subset_id, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder,
                latent_result=latent_result, **model_kwargs
            )

            return self.beam_search(
                target,
                cr,
                subset_id,
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                latent_result=latent_result,
                **model_kwargs,
            )

        else:
            print("not greedy or beam_gen_mode!")


    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        cr: torch.FloatTensor,
        subset_id: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        latent_result: Pack = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)
        cr = cr.index_select(0, expanded_return_idx)
        subset_id = subset_id.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
            
            if latent_result is not None:
                latent_result.sample_z = latent_result.sample_z.index_select(0, expanded_return_idx)

        return input_ids, cr, subset_id, latent_result, model_kwargs


    def beam_search(
        self,
        target,
        cr,
        subset_id,
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
        synced_gpus: Optional[bool] = None,
        latent_result: Optional[Pack] = None,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using beam search decoding.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (:obj:`BeamScorer`):
                An derived instance of :class:`~transformers.BeamScorer` that defines how beam hypotheses are
                constructed, stored and sorted during generation. For more information, the documentation of
                :class:`~transformers.BeamScorer` should be read.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            stopping_criteria (:obj:`StoppingCriteriaList`, `optional`):
                An instance of :class:`~transformers.StoppingCriteriaList`. List of instances of class derived from
                :class:`~transformers.StoppingCriteria` used to tell if the generation loop should stop.
            max_length (:obj:`int`, `optional`, defaults to 20):
                **DEPRECATED**. Use :obj:`logits_processor` or :obj:`stopping_criteria` directly to cap the number of
                generated tokens. The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            synced_gpus (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If
                model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :class:`~transformers.generation_utilsBeamSearchDecoderOnlyOutput`,
            :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput` or obj:`torch.LongTensor`: A
            :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
            :class:`~transformers.generation_utils.BeamSearchDecoderOnlyOutput` if
            ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
            :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput` if
            ``model.config.is_encoder_decoder=True``.


        Examples::

            >>> from transformers import (
            ...    AutoTokenizer,
            ...    AutoModelForSeq2SeqLM,
            ...    LogitsProcessorList,
            ...    MinLengthLogitsProcessor,
            ...    BeamSearchScorer,
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
            ...     "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
            ... }

            >>> # instantiate beam scorer
            >>> beam_scorer = BeamSearchScorer(
            ...     batch_size=1,
            ...     num_beams=num_beams,
            ...     device=model.device,
            ... )

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
            ... ])

            >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
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
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True: # generate next_tokens

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            #
            model_inputs = self.prepare_inputs_for_generation(target, cr, subset_id, input_ids, latent_result=latent_result, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            
            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            # next_token_logits = outputs.logits[:, -1, :]
            if target == "prediction":
                next_token_logits = outputs.prediction_logits[:, -1, :]
            elif target == "reconstruction":
                next_token_logits = outputs.reconstruction_logits[:, -1, :]
                
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            # outputs.decoder_hidden_states should be classify according to target
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    if target == "prediction":
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )
                    elif target == "reconstruction":
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states_for_x,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

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

            # concat as decoder_input_ids of next step
            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation( # update past_key_values
                target, outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if target == "prediction":
                if model_kwargs["past"] is not None:
                    model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)
            elif target == "reconstruction":
                if model_kwargs["past_for_x"] is not None:
                    model_kwargs["past_for_x"] = self._reorder_cache(model_kwargs["past_for_x"], beam_idx)

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
                return (BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                ),
                    input_ids)
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


    def prepare_inputs_for_generation(
        self,
        target,
        cr,
        subset_id,
        decoder_input_ids,
        past=None,
        past_for_x=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        latent_result=None,
        **kwargs
    ):
        decoder_input_ids_for_y = None
        decoder_input_ids_for_x = None

        # cut decoder_input_ids if past is used
        if target == "prediction":
            if past is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            decoder_input_ids_for_y=decoder_input_ids

        elif target == "reconstruction":
            if past_for_x is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            decoder_input_ids_for_x=decoder_input_ids

        return {
            "target": target,
            "cr": cr,
            "subset_id": subset_id,
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "past_key_values_for_x": past_for_x,
            "decoder_input_ids": decoder_input_ids_for_y,
            "decoder_input_ids_for_x": decoder_input_ids_for_x,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "latent_result": latent_result,
        }


    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    @staticmethod
    def _update_model_kwargs_for_generation(
        target, outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        if target == "prediction":
            # update past
            if "past_key_values" in outputs:
                model_kwargs["past"] = outputs.past_key_values
            elif "mems" in outputs:
                model_kwargs["past"] = outputs.mems
            elif "past_buckets_states" in outputs:
                model_kwargs["past"] = outputs.past_buckets_states
            else:
                model_kwargs["past"] = None
        elif target == "reconstruction":
            if "past_key_values_for_x" in outputs:
                model_kwargs["past_for_x"] = outputs.past_key_values_for_x
            else:
                model_kwargs["past_for_x"] = None            

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return model_kwargs