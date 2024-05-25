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


def _forward_pass(model, model_jax, inputs, inputs_jax):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    key = jax.random.PRNGKey(0)
    params = torch_to_jax_states(model, dtype=torch.float32, head_dim=model.config.hidden_size // model.config.num_attention_heads)
    outputs_jax = model_jax.apply(
        params,
        inputs_jax["input_ids"],
        attention_mask=inputs_jax["attention_mask"],
        mutable=("cache",),
        output_hidden_states=True,
    )
    return outputs, outputs_jax, params


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
    model = model_cls(config)
    model_jax = model_cls_jax(config)
    if jit:
        model_jax.apply = jax.jit(
            model_jax.apply,
            static_argnames=["mutable", "output_hidden_states", "use_cache"],
        )
    inputs = tokenizer(["Hello, my dog is cute"] * repeat, return_tensors="pt")
    inputs_jax = tokenizer(["Hello, my dog is cute"] * repeat, return_tensors="jax")
    return tokenizer, model, model_jax, inputs, inputs_jax


@pytest.mark.parametrize("with_mask", [True, False])
def test_model(with_mask: bool):
    # If not computed on CPU, the difference between pytorch is fairly large.
    with jax.default_device(jax.devices("cpu")[0]):
        tokenizer, model, model_jax, inputs, inputs_jax = _setup_models(
            MistralModel, MistralModelJax, jit=False
        )
        if with_mask:
            # With attention mask
            inputs = {
                **inputs,
                "attention_mask": torch.tensor([[0, 0, 0, 1, 1, 1, 1]], dtype=torch.int32),
            }
            inputs_jax = {
                **inputs_jax,
                "attention_mask": jax.numpy.array(
                    [[0, 0, 0, 1, 1, 1, 1]],
                    dtype=jax.numpy.int32,
                ),
            }
        outputs, outputs_jax, _ = _forward_pass(model, model_jax, inputs, inputs_jax)

        for i in range(len(outputs.hidden_states)):
            hidden = outputs.hidden_states[i].numpy()
            hidden_jax = outputs_jax[0].hidden_states[i]
            if with_mask:
                hidden = hidden[:, -4:]
                hidden_jax = hidden_jax[:, -4:]
            print(jnp.max(jnp.abs(hidden - hidden_jax)))
            print(jnp.mean(jnp.abs(hidden_jax)))
            # todo: after a while with new versions of everything, precisions don't work any more... fix it?
            assert jax.numpy.allclose(hidden, hidden_jax, atol=1e-5)


def test_generate():
    with jax.default_device(jax.devices("cpu")[0]):
        tokenizer, model, model_jax, inputs, inputs_jax = _setup_models(
            MistralForCausalLM, MistralForCausalLMJax, repeat=2, jit=True
        )
        outputs, outputs_jax, params = _forward_pass(model, model_jax, inputs, inputs_jax)
        print((outputs.logits.numpy() - outputs_jax[0].logits).max())
        assert jax.numpy.allclose(outputs.logits.numpy(), outputs_jax[0].logits, atol=1e-5)

    # make sure do_sample works and compare to a naive high-latency implementation
    def eval_fn(params, tok, past_key_values=None, use_cache=True) -> tuple:
        out = model_jax.apply(
            params,
            jnp.array(tok),
            mutable=("cache",),
            output_hidden_states=False,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )[0]
        return out.logits, out.past_key_values

    out_jax = model_jax.generate(
        params, inputs_jax["input_ids"], do_sample=True, max_length=20
    )
    naive_out = naive_generate(params, eval_fn, inputs_jax["input_ids"], do_sample=True, max_len=20,)
    assert jnp.all(out_jax == naive_out)
    
    # compare do_sample=False with reference impl
    with torch.no_grad():
        out = model.generate(inputs["input_ids"], do_sample=False, max_new_tokens=10)
    out_jax = model_jax.generate(
        params, inputs_jax["input_ids"], do_sample=False, max_length=10
    )
    #out_jax = naive_generate(params, eval_fn, inputs_jax["input_ids"], do_sample=False, max_len=10)
    outout_jax = naive_generate(params, eval_fn, out_jax[:, :9], do_sample=False, max_len=8)
    #print(out, out_jax)
    #print(outout_jax)

    assert jnp.all(out.numpy() == out_jax)
