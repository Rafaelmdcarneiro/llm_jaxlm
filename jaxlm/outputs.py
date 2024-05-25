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

from typing import Any, List, Optional, Tuple

import chex
import flax
import flax.linen as nn
import jax.numpy as jnp


@flax.struct.dataclass
class BaseModelOutputWithPast:
    last_hidden_state: jnp.ndarray
    past_key_values: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
    hidden_states: Optional[Tuple[jnp.ndarray, ...]] = None
    attentions: Optional[Tuple[jnp.ndarray, ...]] = None


@flax.struct.dataclass
class CausalLMOutputWithPast:
    logits: jnp.ndarray
    past_key_values: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
    hidden_states: Optional[Tuple[jnp.ndarray, ...]] = None
    attentions: Optional[Tuple[jnp.ndarray, ...]] = None
