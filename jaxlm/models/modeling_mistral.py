# coding=utf-8
# Copyright 2023 Honglu Fan (https://github.com/honglu2875).
#
# This code is based on Hugging Face Mistral model code whose authors are
# denoted below. But it has been largely modified for JAX, Flax, and t5x.
# Original copyright message below:
#
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
import functools
import warnings
from functools import partial
from typing import Any, List, Optional, Tuple

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import partitioning as nn_partitioning
from flax.linen.partitioning import param_with_axes, with_sharding_constraint
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from t5x import partitioning as t5x_partitioning
from t5x.examples.t5 import layers

from .._generate import generate
from ..nn.attention import Attention
from ..nn.linear import DenseGeneral
from ..nn.norms import RMSNorm
from ..nn.embedding import Embed
from ..outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..nn.position import RotaryEmbedding, apply_rotary_pos_emb
from ..types import Array
from ..utils import check_shape

"""
Notes:
It uses t5x.examples.t5.layers so that it is compatible with the t5 library. But t5x defines logical named axis
and operates sharding in a different fashion than the flax official way of using `nn.with_logical_partitioning`...
Although I hate doing this I am mixing both ways. 
Putting this note out there to say that it's not my fault for this code. For any serious use, this needs a lot 
of refactoring. I have already cleaned up some mess from the insane Hugging Face `modeling_mistral.py` and
hopefully things are not too difficult from here.
"""


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
@jax.jit
def _get_unpad_data(padding_mask):
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=jnp.int32)
    indices = jnp.nonzero(padding_mask.flatten(), as_tuple=False)[0].flatten()
    max_seqlen_in_batch = seqlens_in_batch.max()
    cu_seqlens = jnp.pad(jnp.cumsum(seqlens_in_batch, dim=0, dtype=jnp.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


@partial(
    jax.jit,
    static_argnames=(
        "input_ids_shape",
        "dtype",
        "past_key_values_length",
        "sliding_window",
    ),
)
def _make_sliding_window_causal_mask(
    input_ids_shape: tuple,
    dtype: jnp.dtype,
    past_key_values_length: int = 0,
    sliding_window: int = 4096,
):
    """
    Make causal mask used for sliding window attention
    """
    bsz, tgt_len = input_ids_shape

    if tgt_len == 1 and past_key_values_length > 0:
        # we are likely at inferencing stage and the causal mask can be fast-tracked
        pad_len = tgt_len + past_key_values_length - sliding_window - 1
        return jnp.log(
            jnp.triu(jnp.ones((1, tgt_len + past_key_values_length)), k=pad_len)
        )

    tensor = jnp.ones(
        (tgt_len, tgt_len),
    )
    mask = jnp.tril(tensor, k=0)
    # make the mask banded to account for sliding window
    mask = jnp.triu(mask, k=-sliding_window)
    mask = jnp.log(mask).astype(dtype)

    if past_key_values_length > 0:
        mask = jnp.concatenate(
            [jnp.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1
        )
    return jnp.broadcast_to(
        mask[None, None, :, :], (bsz, 1, tgt_len, tgt_len + past_key_values_length)
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
@partial(jax.jit, static_argnums=(1,), static_argnames=("tgt_len",))
def _expand_mask(mask: jnp.ndarray, dtype: jnp.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = jnp.broadcast_to(
        mask[:, None, None, :], (bsz, 1, tgt_len, src_len)
    ).astype(dtype)

    return jnp.where(expanded_mask == 0, jnp.finfo(dtype).min, 0.0).astype(dtype)


def repeat_kv(hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = jnp.broadcast_to(
        hidden_states[:, :, :, None, :],
        (batch, slen, num_key_value_heads, n_rep, head_dim),
    )
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


class MistralMLP(nn.Module):
    config: Any = None
    dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform
    act_fn: Any = jax.nn.silu

    def setup(self):
        if self.config is None:
            raise ValueError("Must provide a config for MLP.")
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size
        # input dim supposed to be self.hidden_size
        self.gate_proj = DenseGeneral(
            features=self.intermediate_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_init_args=(),
            with_logical_partitioning=True,
            kernel_axes=("embed", "intermediate"),
            name="gate_proj",
        )
        self.up_proj = DenseGeneral(
            features=self.intermediate_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_init_args=(),
            with_logical_partitioning=True,
            kernel_axes=("embed", "intermediate"),
            name="up_proj",
        )
        # input dim supposed to be self.intermediate_size
        self.down_proj = DenseGeneral(
            features=self.hidden_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_init_args=(),
            with_logical_partitioning=True,
            kernel_axes=("intermediate", "embed"),
            name="down_proj",
        )

    def __call__(self, x, training=False):
        assert (
            x.shape[-1] == self.hidden_size
        ), f"Input to MLP layers have different dimensions than the hidden dimension. Got {x.shape[-1]}"
        x = with_sharding_constraint(x, ("batch", "length", "embed"))
        gate = self.act_fn(self.gate_proj(x))
        proj = self.up_proj(x)
        x = self.down_proj(gate * proj)
        return x


class MistralAttention(Attention):
    def setup(self):
        if self.config is None:
            raise ValueError("Must provide a config for attention.")

        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        if self.fused_qkv:
            assert self.num_heads == self.num_key_value_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_length=self.max_position_embeddings,
            base=self.rope_theta,
        )

        super().setup()

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        padding_mask=None,
        training=False,
    ) -> tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[tuple]]:
        assert (
            hidden_states.shape[-1] == self.hidden_size
        ), f"Input to Attention layer has different dimension than the hidden dimension. Got {hidden_states.shape[-1]}"
        bsz, q_len = hidden_states.shape[-3:-1]  # bsz, q_len, hidden_size

        # Obtain q, k, v from the current hidden state and shard q only (k, v will be handled later)
        query_states, key_states, value_states = self.qkv_proj(
            hidden_states
        )  # bsz, seq, n_head, head_dim

        past_kv_length = (
            past_key_value[0].shape[-3] if past_key_value is not None else 0
        )
        kv_seq_len = key_states.shape[-3] + past_kv_length
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids[:, past_kv_length:]
        )
        # attach kv-cache to k and v if exists, and shard k, v accordingly
        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), "past_key_value should be a tuple of (k, v)"
            past_key, past_value = past_key_value
            key_states = jnp.concatenate([past_key, key_states], axis=1)
            value_states = jnp.concatenate([past_value, value_states], axis=1)

        if use_cache:
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = jnp.einsum(
            "bshn,bthn->bhst", query_states, key_states
        ) / jnp.sqrt(self.head_dim)

        check_shape(attn_weights, bsz, self.num_heads, q_len, kv_seq_len)

        if attention_mask is not None:
            check_shape(attention_mask, bsz, 1, q_len, kv_seq_len)
            attn_weights = attn_weights + attention_mask

        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(
            hidden_states.dtype
        )

        attn_output = jnp.einsum("bhst,bthn->bshn", attn_weights, value_states)
        check_shape(attn_output, bsz, q_len, self.num_heads, self.head_dim)

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MistralDecoderLayer(nn.Module):
    config: Any = None
    dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.self_attn = MistralAttention(
            config=self.config, dtype=self.dtype, kernel_init=self.kernel_init
        )
        self.mlp = MistralMLP(
            config=self.config, dtype=self.dtype, kernel_init=self.kernel_init
        )
        self.input_layernorm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple:
        """
        Args:
            hidden_states: input tensor
            attention_mask: mask for attention layer
            position_ids: position ids for positional embeddings
            past_key_value: cached key and value projection states
            output_attentions: whether to output attention weights
            use_cache: whether to use cached key and value projection states
            padding_mask: mask for padding
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MistralModel(nn.Module):
    config: Any = None
    dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform

    def setup(self):
        self.padding_idx = self.config.pad_token_id
        self.vocab_size = self.config.vocab_size

        self.embed_tokens = Embed(
            num_embeddings=self.vocab_size,
            features=self.config.hidden_size,
            kernel_init=self.kernel_init,
            kernel_init_args=(),
            one_hot=True,
            name="embed_tokens",
        )
        self.layers = [
            MistralDecoderLayer(
                self.config, dtype=self.dtype, kernel_init=self.kernel_init
            )
            for _ in range(self.config.num_hidden_layers)
        ]
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    @staticmethod
    def _prepare_decoder_attention_mask(
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length,
        sliding_window,
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = _make_sliding_window_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            past_key_values_length=past_key_values_length,
            sliding_window=sliding_window,
        )

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        )

        return expanded_attn_mask + combined_attention_mask

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[List[jnp.ndarray]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        batch_size, seq_length = input_ids.shape

        past_key_values_length = (
            0 if past_key_values is None else past_key_values[0][0].shape[1]
        )

        padding_mask = None

        # embed positions
        if attention_mask is None:
            attention_mask = jnp.ones(
                (batch_size, seq_length + past_key_values_length), dtype=bool
            )
        else:
            padding_mask = attention_mask

        if position_ids is None:
            position_ids = (
                jnp.arange(
                    seq_length + past_key_values_length,
                    dtype=jnp.int32,
                )[None]
                - (~attention_mask[0, :-seq_length]).sum()
            )
            position_ids = jnp.where(position_ids >= 0, position_ids, 0)

        inputs_embeds = self.embed_tokens(input_ids).astype(self.dtype)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

        hidden_states = with_sharding_constraint(
            inputs_embeds, ("batch", "length", "embed")
        )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MistralForCausalLM(nn.Module):
    config: Any = None
    dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.xavier_uniform

    sharded: Optional[bool] = len(jax.devices()) > 1 and len(jax.devices()) % 2 == 0

    @staticmethod
    def mesh_sharding(pspec: PartitionSpec | None, mesh: Mesh | None) -> NamedSharding:
        if mesh is None:
            mesh = Mesh(jax.devices(), (None,))
        return NamedSharding(mesh, pspec)

    @staticmethod
    def _parse_mesh_layout(device_mesh_layout):
        assert isinstance(device_mesh_layout, (list, tuple)), (
            f"device_mesh_layout must be a list or tuple. "
            f"Got {type(device_mesh_layout)}"
        )
        assert len(device_mesh_layout) == 2, (
            f"The length of device_mesh_layout must be 2. "
            f"Got {len(device_mesh_layout)}"
        )
        mesh_layout = []
        for i in range(2):
            if device_mesh_layout[i] is None:
                assert (
                    device_mesh_layout[1 - i] is not None
                ), f"Invalid device_mesh_layout. Got {device_mesh_layout}."
                mesh_layout.append(len(jax.devices()) // device_mesh_layout[1 - i])
            else:
                mesh_layout.append(device_mesh_layout[i])

        return tuple(mesh_layout)

    def _shard_params(self, x, y):
        if x.ndim != len(y.spec):
            assert (
                x.ndim == 2 and len(y.spec) == 3
            ), f"The shape of x ({x.shape}) and the sharding spec ({y.spec}) does not match"
            warnings.warn(
                f"The parameter has 2 axis ({x.shape}) while the sharding spec ({y.spec}) has 3 axis. "
                "Attempting to reshape into [:, :, head_dim], but please confirm that this is the intended behavior."
            )
            return jax.device_put(
                x.reshape(
                    (
                        x.shape[0],
                        -1,
                        self.config.hidden_size // self.config.num_attention_heads,
                    )
                ),
                y,
            )
        return (jax.device_put(x, y),)

    def get_params(self, device_mesh_layout=(1, None), weights=None):
        """
        Get the properly sharded parameters.
        Args:
            device_mesh_layout: the device mesh layout. For example:
                (1, None) means data=1, model=len(jax.devices())
                (2, None) means data=2, model=len(jax.devices()) // 2
                (None, 2) means data=len(jax.devices()) // 2, model=2
            weights: whether a tree of weights are already given (but may not be sharded)
        Returns:
            a tree of properly sharded parameters
        """
        key = jax.random.PRNGKey(0)

        mesh_layout = self._parse_mesh_layout(device_mesh_layout)

        dummy_input = jnp.array(
            [[1 for _ in range(mesh_layout[1])] for _ in range(mesh_layout[0])]
        )

        abstract_variables = jax.eval_shape(self.init, key, dummy_input)
        if self.sharded:
            mesh = Mesh(
                devices=mesh_utils.create_device_mesh(mesh_layout),
                axis_names=("data", "model"),
            )

            rules = t5x_partitioning.standard_logical_axis_rules(
                activation_partitioning_dims=1,
                parameter_partitioning_dims=1,
                additional_rules=(
                    ("kv_length", None),
                    ("intermediate", "model"),
                ),
            )
            logical_state_spec = nn.get_partition_spec(abstract_variables)
            logical_state_sharding = nn.logical_to_mesh_sharding(
                logical_state_spec, mesh, rules
            )

            x_sharding = self.mesh_sharding(
                PartitionSpec("data", None), mesh
            )  # dimensions: (batch, length)

            if weights is not None:
                assert isinstance(
                    weights, dict
                ), f"weights must be a dict, got {type(weights)}"
                assert (
                    "params" in weights
                ), f"The key params not found in 'weights'. Got {weights.keys()}"

                if self.sharded:
                    params = {
                        "params": jax.tree_util.tree_map(
                            self._shard_params,
                            weights["params"],
                            logical_state_sharding["params"],
                        )
                    }
                else:
                    params = weights
            else:
                params = jax.jit(
                    self.init,
                    in_shardings=(
                        self.mesh_sharding(None, mesh),
                        x_sharding,
                    ),  # PRNG key and x
                    out_shardings=logical_state_sharding,
                )(key, dummy_input)
        else:
            params = self.init(key, dummy_input)

        return params

    def prepare_input(self, inputs, device_mesh_layout=(1, None), dtype=None):
        if self.sharded:
            mesh = Mesh(
                devices=mesh_utils.create_device_mesh(
                    self._parse_mesh_layout(device_mesh_layout)
                ),
                axis_names=("data", "model"),
            )
            inputs = jax.device_put(
                inputs, self.mesh_sharding(PartitionSpec("data", None), mesh)
            )
        if dtype is not None:
            inputs = jax.tree_util.tree_map(lambda x: x.astype(dtype), inputs)
        return inputs

    def setup(self):
        self.model = MistralModel(
            self.config, dtype=self.dtype, kernel_init=self.kernel_init
        )
        self.lm_head = DenseGeneral(
            features=self.config.vocab_size,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_init_args=(),
            with_logical_partitioning=True,
            kernel_axes=("embed", "vocab"),
            name="lm_head",
        )

    def __call__(
        self,
        input_ids,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[List[jnp.ndarray]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def wrapped_apply_fn(
        self,
        params,
        tok,
        past_key_values=None,
        use_cache=True,
        unpadded_past_kv_length=None,
    ) -> tuple[CausalLMOutputWithPast, dict]:
        tok = jnp.array(tok)
        if unpadded_past_kv_length is not None:
            assert past_key_values is not None
            assert tok.shape[1] == 1, (
                f"When kv cache padding is enabled, only support when query seq = 1. "
                f"Got {tok.shape[1]} instead."
            )
            past_kv_length = past_key_values[0][0].shape[1]

            position_ids = jnp.arange(past_kv_length + 1) - (
                past_kv_length - unpadded_past_kv_length
            )
            attention_mask = jnp.repeat(
                (position_ids >= 0)[None], repeats=tok.shape[0], axis=0
            )
            position_ids = jnp.repeat(
                jnp.where(position_ids >= 0, position_ids, 0)[None],
                repeats=tok.shape[0],
                axis=0,
            )
        else:
            position_ids, attention_mask = None, None

        out = self.apply(
            params,
            tok,
            position_ids=position_ids,
            attention_mask=attention_mask,
            mutable=("cache",),
            output_hidden_states=False,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )[
            0
        ]  # return a tuple (CausalLMOutputWithPast, dict) where dict is the mutable cache
        if (
            unpadded_past_kv_length is not None
        ):  # padding is applied: truncate the past kv back to same length
            past_key_values = jax.tree_util.tree_map(
                lambda x: x[:, tok.shape[1] :], out.past_key_values
            )
        else:
            past_key_values = out.past_key_values
        return out.logits, past_key_values

    def generate(
        self,
        params,
        prompt_tokens: list | jnp.ndarray,
        do_sample: bool = True,
        seed: int = 0,
        max_length: int = 10,
        top_k: int = 0,
        top_p: float = 0.0,
        temp: float = 1.0,
        no_jit: bool = False,
    ):
        if no_jit:
            apply = self.wrapped_apply_fn
        else:
            apply = jax.jit(self.wrapped_apply_fn, static_argnames=("use_cache",))

        return generate(
            params,
            apply,
            prompt_tokens,
            do_sample=do_sample,
            seed=seed,
            max_len=max_length,
            top_k=top_k,
            top_p=top_p,
            temp=temp,
        )
