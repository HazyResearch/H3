# Copyright (c) 2023, Tri Dao, Dan Fu.

import math
import re
from functools import partial

from collections import namedtuple, OrderedDict
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import Mlp, FusedMLP
from flash_attn.modules.block import Block
from flash_attn.modules.embedding import GPT2Embeddings
from flash_attn.utils.generation import GenerationMixin

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

from src.models.ssm.h3 import H3


def create_mixer_cls(ssm_cls=H3, ssm_cfg=None, attn_layer_idx=None, attn_cfg=None, layer_idx=None):
    if attn_layer_idx is not None and layer_idx in attn_layer_idx:
        causal = True if attn_cfg is None else attn_cfg.pop('causal', True)
        mixer_cls = partial(MHA, layer_idx=layer_idx, causal=causal, 
                            **(attn_cfg if attn_cfg is not None else {}))
    else:
        mixer_cls = partial(ssm_cls, layer_idx=layer_idx,
                            **(ssm_cfg if ssm_cfg is not None else {}))
    return mixer_cls


def create_mlp_cls(d_model, d_inner=None, fused_mlp=False):
    inner_dim = d_inner if d_inner is not None else 4 * d_model
    if not fused_mlp:
        mlp_cls = partial(Mlp, hidden_features=inner_dim,
                          activation=partial(F.gelu, approximate='tanh'))
    else:
        mlp_cls = partial(FusedMLP, hidden_features=inner_dim)
    return mlp_cls


def create_block(d_model, d_inner=None, ssm_cls=H3, ssm_cfg=None, attn_layer_idx=None,
                 attn_cfg=None, layer_norm_epsilon=1e-5,
                 resid_dropout1=0.0, resid_dropout2=0.0, residual_in_fp32=False,
                 fused_mlp=False, fused_dropout_add_ln=False, layer_idx=None):
    mixer_cls = create_mixer_cls(ssm_cls=ssm_cls, ssm_cfg=ssm_cfg, attn_layer_idx=attn_layer_idx,
                                 attn_cfg=attn_cfg, layer_idx=layer_idx)
    mlp_cls = create_mlp_cls(d_model, d_inner=d_inner, fused_mlp=fused_mlp)
    norm_cls = partial(nn.LayerNorm, eps=layer_norm_epsilon)
    block = Block(d_model, mixer_cls, mlp_cls, norm_cls=norm_cls,
                  prenorm=True, resid_dropout1=resid_dropout1, resid_dropout2=resid_dropout2,
                  fused_dropout_add_ln=fused_dropout_add_ln, residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True,
                  glu_act=False):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
            # If using GLU activation for now, we scale the std by 2
            elif name in ["output_linear.0.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                if not glu_act:
                    nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
                else:
                    out_features = p.shape[0]
                    # Multiplying the first half of the matrix by 2 since sigmoid scales it down by 0.5
                    # on average.
                    nn.init.normal_(p[:out_features // 2], mean=0.0, std=initializer_range / math.sqrt(2 * n_layer) * 2)


class SSMModel(nn.Module):

    def __init__(self, d_model: int, n_layer: int, d_inner: int, vocab_size: int, ssm_cfg=None,
                 attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0,
                 resid_dropout: float = 0.0, embed_dropout: float = 0.1, dropout_cls=nn.Dropout,
                 layer_norm_epsilon: float = 1e-5, initializer_cfg=None,
                 fused_mlp=False, fused_dropout_add_ln=False, residual_in_fp32=False,
                 **kwargs) -> None:
        super().__init__()
        self.embeddings = GPT2Embeddings(d_model, vocab_size, max_position_embeddings)
        self.residual_in_fp32 = residual_in_fp32

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.fused_dropout_add_ln = fused_dropout_add_ln
        if self.fused_dropout_add_ln and dropout_add_layer_norm is None:
            raise ImportError('dropout_add_layer_norm is not installed')

        self.layers = nn.ModuleList([create_block(
            d_model, d_inner=d_inner, ssm_cfg=ssm_cfg, attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg, layer_norm_epsilon=layer_norm_epsilon,
            resid_dropout1=embed_dropout if i == 0 else resid_dropout,
            resid_dropout2=resid_dropout, residual_in_fp32=residual_in_fp32,
            fused_mlp=fused_mlp, fused_dropout_add_ln=fused_dropout_add_ln, layer_idx=i
        ) for i in range(n_layer)])

        self.drop_f = nn.Dropout(resid_dropout)
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg if initializer_cfg is not None else {})))

    def forward(self, input_ids, position_ids=None, inference_params=None):
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        residual = None
        mixer_kwargs = None
        if inference_params is not None:
            mixer_kwargs = dict(inference_params=inference_params)
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, mixer_kwargs=mixer_kwargs)
        if not self.fused_dropout_add_ln:
            dropped = self.drop_f(hidden_states)
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = dropout_add_layer_norm(
                hidden_states, residual, self.ln_f.weight, self.ln_f.bias,
                self.drop_f.p if self.training else 0.0, self.ln_f.eps, prenorm=False,
                residual_in_fp32=self.residual_in_fp32
            )
        return hidden_states


class SSMLMHeadModel(nn.Module, GenerationMixin):

    def __init__(self, d_model: int, n_layer: int, d_inner: int, vocab_size: int, ssm_cfg=None,
                 attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0,
                 resid_dropout: float = 0.0, embed_dropout: float = 0.1, dropout_cls=nn.Dropout,
                 layer_norm_epsilon: float = 1e-5, initializer_cfg=None,
                 fused_mlp=False, fused_dropout_add_ln=False, residual_in_fp32=False,
                 pad_vocab_size_multiple: int = 1, **kwargs) -> None:
        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = SSMModel(
            d_model=d_model, n_layer=n_layer, d_inner=d_inner, vocab_size=vocab_size,
            ssm_cfg=ssm_cfg, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout, embed_dropout=embed_dropout,
            dropout_cls=dropout_cls, layer_norm_epsilon=layer_norm_epsilon,
            initializer_cfg=initializer_cfg, fused_mlp=fused_mlp,
            fused_dropout_add_ln=fused_dropout_add_ln, residual_in_fp32=residual_in_fp32, **kwargs
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg if initializer_cfg is not None else {})))
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight

    def forward(self, input_ids, position_ids=None, inference_params=None):
        hidden_states = self.backbone(input_ids, position_ids=position_ids,
                                      inference_params=inference_params)
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
        return CausalLMOutput(logits=lm_logits)

    def load_state_dict(self, state_dict, strict=True):
        # Remapping from our checkpoints that used different names
        def key_mapping_backbone(key):
            key = re.sub(r'^s4seq.encoder.', 'backbone.', key)
            key = re.sub(r'^embedding.', 'backbone.embeddings.word_embeddings.', key)
            key = re.sub(r'^backbone.norm', 'backbone.ln_0', key)
            return key
        state_dict = OrderedDict((key_mapping_backbone(k), v) for k, v in state_dict.items())
        # Remapping from our checkpoints that used a different ordering of layers in the block
        # Previous: Mixer / MLP -> Dropout -> Add -> LN
        # Current: Dropout -> Add -> LN -> Attn / MLP
        if 'backbone.ln_0.weight' in state_dict:
            n_layers = len(self.backbone.layers)
            ln_weight = state_dict.pop(f'backbone.layers.{n_layers - 1}.norm2.weight')
            ln_bias = state_dict.pop(f'backbone.layers.{n_layers - 1}.norm2.bias')
            state_dict['backbone.ln_f.weight'] = ln_weight
            state_dict['backbone.ln_f.bias'] = ln_bias
            for l in reversed(range(n_layers)):
                ln_weight = state_dict.pop(f'backbone.layers.{l}.norm1.weight')
                ln_bias = state_dict.pop(f'backbone.layers.{l}.norm1.bias')
                state_dict[f'backbone.layers.{l}.norm2.weight'] = ln_weight
                state_dict[f'backbone.layers.{l}.norm2.bias'] = ln_bias
                if l > 0:
                    ln_weight = state_dict.pop(f'backbone.layers.{l - 1}.norm2.weight')
                    ln_bias = state_dict.pop(f'backbone.layers.{l - 1}.norm2.bias')
                    state_dict[f'backbone.layers.{l}.norm1.weight'] = ln_weight
                    state_dict[f'backbone.layers.{l}.norm1.bias'] = ln_bias
            ln_weight = state_dict.pop('backbone.ln_0.weight')
            ln_bias = state_dict.pop('backbone.ln_0.bias')
            state_dict[f'backbone.layers.0.norm1.weight'] = ln_weight
            state_dict[f'backbone.layers.0.norm1.bias'] = ln_bias
        return super().load_state_dict(state_dict, strict=strict)
