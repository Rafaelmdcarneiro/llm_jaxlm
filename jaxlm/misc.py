# coding=utf-8
# Copyright 2023 Honglu Fan (https://github.com/honglu2875).
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

import jax
import jax.numpy as jnp


def _get_causal_mask(qs, ks):
    assert ks >= qs, "key_len must be greater than or equal to query_len"
    return jnp.triu(jnp.ones((qs, ks), dtype=bool), ks - qs + 1)


def _attn_fn(
    query, key, value, past_position=0, causal=True
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Eager implementation of attention. c.f. `attention_ref(...)` in
    `https://github.com/HazyResearch/flash-attention/blob/main/tests/test_flash_attn.py`.
    It is still off by a minor amount when comparing with the flash attention, but should not be practically
    a problem.

    query: [..., query_len, num_heads, head_dim]
    key: [..., key_len, num_heads, head_dim]
    value: [..., value_len, num_heads, head_dim]
    (if pmap/pjit is applied, ... could have 2 dims involving data-parallel in addition to batch dim)

    key_len is normally the same as value_len, but could possibly larger than query_len (implementing kv cache).
    """
    attn_weights = jnp.einsum(
        "...shd,...thd->...hst", query / jnp.sqrt(query.shape[-1]), key
    )
    query_len = query.shape[-3]
    key_len = key.shape[-3]
    attn_weights = jnp.where(
        causal,
        jnp.where(
            ~_get_causal_mask(query.shape[-3], key.shape[-3]),
            attn_weights,
            float("-inf"),
        ),
        attn_weights,
    )
    mask = jnp.arange(0, key_len, dtype=jnp.int32) < (
        key_len - past_position - query_len
    )
    mask_shape = (1,) * (attn_weights.ndim - 1) + (-1,)
    attn_weights = jnp.where(mask.reshape(mask_shape), float("-inf"), attn_weights)
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    attn_output = jnp.einsum("...hst,...thd->...shd", attn_weights, value)
    return attn_output, attn_weights


# TODO: implement flash attention in JAX and compare with the above as unit tests
