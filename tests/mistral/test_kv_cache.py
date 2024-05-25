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
import pytest        
import torch
from transformers import (AutoTokenizer, MistralConfig, MistralForCausalLM,
                          MistralModel)

from jaxlm import MistralForCausalLM as MistralForCausalLMJax
from jaxlm import MistralModel as MistralModelJax
from jaxlm.utils import torch_to_jax_states
from jaxlm.test_utils.naive_generate import generate as naive_generate


def _forward_pass(model_jax, inputs_jax, use_cache=False):
    key = jax.random.PRNGKey(0)
    #params = torch_to_jax_states(model, dtype=torch.float32, head_dim=model.config.hidden_size // model.config.num_attention_heads)
    params = model_jax.init(
        key,
        inputs_jax["input_ids"],
        #attention_mask=inputs_jax["attention_mask"],
        #mutable=("cache",),
        #output_hidden_states=True,
        #use_cache=use_cache,
    )
    outputs_jax, _ = model_jax.apply(
        params,
        inputs_jax["input_ids"],
        attention_mask=inputs_jax["attention_mask"],
        mutable=("cache",),
        output_hidden_states=True,
        use_cache=use_cache,
    )
    return outputs_jax, params


def _setup_models(model_cls, model_cls_jax, jit=True, repeat=1):
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Yarn-Mistral-7b-64k")
    config = MistralConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=2,
        sliding_window=3,
    )
    #model = model_cls(config)
    model_jax = model_cls_jax(config)
    if jit:
        model_jax.apply = jax.jit(
            model_jax.apply,
            static_argnames=["mutable", "output_hidden_states", "use_cache"],
        )
    #inputs = tokenizer(["Hello, my dog is cute"] * repeat, return_tensors="pt")
    inputs_jax = tokenizer(["Hello, my dog is cute"] * repeat, return_tensors="jax")
    return tokenizer, model_jax, inputs_jax


def test_kv_cache():
    with jax.default_device(jax.devices("cpu")[0]):
        tokenizer, model_jax, inputs_jax = _setup_models(
            MistralModel, MistralModelJax, jit=False
        )

        outputs_jax, params = _forward_pass(model_jax, inputs_jax, use_cache=True)

        #out, past_kv = outputs_jax
        out = outputs_jax.last_hidden_state
        past_kv = outputs_jax.past_key_values

        outputs_jax2, _ = model_jax.apply(
            params,
            inputs_jax["input_ids"][:, -1:],
            attention_mask=None,
            mutable=("cache",),
            output_hidden_states=True,
            use_cache=True,
            past_key_values=[(x[0][:, :-1], x[1][:, :-1]) for x in past_kv],
        )
        out2 = outputs_jax2.last_hidden_state
        past_kv2 = outputs_jax2.past_key_values

        for (k,v), (k2,v2) in zip(past_kv, past_kv2):
            #print(jnp.abs(k[:, :-1]-k2[:, :-1]).max())
            #print(jnp.abs(v[:, :-1]-v2[:, :-1]).max())
            print(jnp.abs(k-k2).max())
            print(jnp.abs(v-v2).max())
        print(jnp.abs(out[:, -1:] - out2).max())
        assert jnp.allclose(out[:, -1:], out2, atol=1e-5)

