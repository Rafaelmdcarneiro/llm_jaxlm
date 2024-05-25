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
import flax.linen as nn
from flax import struct
from .types import Array


class KVCache(struct.PyTreeNode):
    """Simple pytree object for recording kv cache."""
    k: Array = struct.field(pytree_node=True)
    v: Array = struct.field(pytree_node=True)
    # kv cache is sometimes padded. end_pos indicate its ending position.
    end_pos: int = -1
    # kv cache may also have padding to the left, and one can apply a mask.
    mask: Array = struct.field(pytree_node=True)

    def get_kv(self):
        if end_pos == -1:
            return self.k, self.v
        return self.k[:end_pos], self.v[:end_pos]

    def get_kv_mask(self):
        if end_pos == -1:



