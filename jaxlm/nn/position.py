#  Copyright 2024 Honglu Fan
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from ..types import DType


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # [seq_len, dim] -> [batch_size, seq_len, 1, head_dim]
    cos = jnp.expand_dims(jnp.take(cos, position_ids, axis=0), axis=2)
    sin = jnp.expand_dims(jnp.take(sin, position_ids, axis=0), axis=2)
    # q_len, k_len = q.shape[1], k.shape[1]
    # q_embed = (q * cos[:, -q_len:]) + (rotate_half(q) * sin[:, -q_len:])
    # k_embed = (k * cos[:, -k_len:]) + (rotate_half(k) * sin[:, -k_len:])
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    dim: int
    # max_trained_length is the initial context window, and we may extend it at inference time.
    max_length: int = 2048
    base: int = 10000
    dtype: DType = jnp.float32
    disable_cache: bool = False

    def setup(self):
        self.inv_freq = self.variable(
            "cache",
            "inv_freq",
            lambda: 1.0
            / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)),
        )

        if not self.disable_cache:
            emb = self.get_emb(self.max_length)
            self.cos_cached = self.variable(
                "cache", "cos_cached", lambda: jnp.cos(emb).astype(self.dtype)
            )
            self.sin_cached = self.variable(
                "cache", "sin_cached", lambda: jnp.sin(emb).astype(self.dtype)
            )

    def get_emb(self, seq_len: int):
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, self.inv_freq.value)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return emb

    def __call__(self, x: jnp.ndarray, seq_len=None):
        # x: [bs, seq_len, num_attention_heads, head_size]
        if seq_len is None:
            seq_len = x.shape[1]

        if self.disable_cache:
            # Skip updating caches and directly go for the result.
            emb = self.get_emb(seq_len)
            return jnp.cos(emb).astype(x.dtype), jnp.sin(emb).astype(x.dtype)

        return (
            self.cos_cached.value[:seq_len].astype(x.dtype),
            self.sin_cached.value[:seq_len].astype(x.dtype),
        )
