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

from typing import Callable

import chex
import numpy as np
import orbax
import torch

from .types import Array, DType, PRNGKey, Shape


def torch_to_jax_states(
    input: torch.nn.Module | dict,
    dtype: str | torch.dtype = torch.float16,
    head_dim: int | None = None,
):
    """
    Converts the states of a PyTorch model to JAX states.
    """
    _to_np_dtype = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        # "bf16": np.float16,
    }

    if isinstance(input, torch.nn.Module):
        states = input.state_dict()
    elif isinstance(input, dict):
        states = input
    else:
        raise TypeError(
            f"Expected input to be either a PyTorch module or a dict, got {type(input)}."
        )

    jax_states = {"params": {}}

    _dense_key_map = {"weight": ("kernel", lambda x: x.T)}
    if head_dim is None:
        _qkv_separate_map = _dense_key_map
    else:
        _qkv_separate_map = {
            "weight": ("kernel", lambda x: x.T.reshape(x.shape[1], -1, head_dim))
        }
    _emb_key_map = {"weight": ("embedding", lambda x: x)}
    _exclude_keys = {"post_attention_layernorm", "input_layernorm", "norm"}

    for k, v in states.items():
        if k.endswith("bias"):
            raise NotImplementedError(
                "Not implemented for bias conversion as Mistral does not use bias."
            )
        split = k.split(".")
        for i, s in enumerate(split):
            if s.isdigit():
                split[i - 1] += "_" + s
                split.pop(i)

        if split[-2] in _exclude_keys:
            _key_map = {}
        else:
            if "embed_tokens" in split:
                _key_map = _emb_key_map
            elif any(k in split for k in ["q_proj", "k_proj", "v_proj"]):
                _key_map = _qkv_separate_map
            else:
                _key_map = _dense_key_map

        if split[-1] in _key_map:
            split[-1], func = _key_map[split[-1]]
            val = func(v.numpy().astype(_to_np_dtype[dtype]))
        else:
            val = v.numpy().astype(_to_np_dtype[dtype])

        _dict = jax_states["params"]
        for i, l in enumerate(split):
            _dict[l] = _dict.setdefault(l, {} if i < len(split) - 1 else val)
            _dict = _dict[l]

    return jax_states


def save(params, path="tmp/"):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_checkpointer.save(path, params)


def load(path="tmp/", item=None):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    return orbax_checkpointer.restore(path, item=item)


def check_shape(tensor, *shape):
    chex.assert_shape(tensor, shape)
