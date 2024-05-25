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
import functools
import os
import pathlib
import shutil

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.linen import partitioning as nn_partitioning
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from t5x import partitioning
from transformers import AutoTokenizer, MistralConfig, MistralForCausalLM

from jaxlm import MistralForCausalLM as MistralForCausalLMJax
from jaxlm.utils import load, save, torch_to_jax_states

num_devices = jax.device_count()
device_mesh = mesh_utils.create_device_mesh((2, num_devices // 2))
mesh = Mesh(devices=device_mesh, axis_names=("data", "model"))
with_sharding_constraint = nn_partitioning.with_sharding_constraint
MODEL_PATH = "NousResearch/Yarn-Mistral-7b-64k"


def mesh_sharding(pspec: PartitionSpec | None) -> NamedSharding:
    return NamedSharding(mesh, pspec)


def test_sharding():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    config = MistralConfig(
        hidden_size=128,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=4,
        sliding_window=3,
    )

    inputs_jax = tokenizer(
        ["Hello, my dog is cute", "Hello, my dog is cute"], return_tensors="jax"
    )

    model_jax = MistralForCausalLMJax(config)

    rules = partitioning.standard_logical_axis_rules(
        activation_partitioning_dims=1,
        parameter_partitioning_dims=1,
        additional_rules=(
            ("kv_length", None),
            ("intermediate", None),
            ("up_sample", "model"),
        ),
    )

    model_jax.apply = jax.jit(
        functools.partial(
            model_jax.apply, mutable=("cache",), output_hidden_states=False
        ),
    )
    key = jax.random.PRNGKey(0)

    abstract_variables = jax.eval_shape(model_jax.init, key, inputs_jax["input_ids"])
    logical_state_spec = nn.get_partition_spec(abstract_variables)
    logical_state_sharding = nn.logical_to_mesh_sharding(
        logical_state_spec, mesh, rules
    )

    x_sharding = mesh_sharding(
        PartitionSpec("data", None)
    )  # dimensions: (batch, length)

    jit_init_fn = jax.jit(
        model_jax.init,
        in_shardings=(mesh_sharding(None), x_sharding),  # PRNG key and x
        out_shardings=logical_state_sharding,
    )
    params = jit_init_fn(key, inputs_jax["input_ids"])

    print("Rules:")
    print(rules)
    print("lm_head kernel sharding:")
    jax.debug.visualize_array_sharding(params["params"]["lm_head"]["kernel"].value)
    print("layer 0 mlp down_proj kernel sharding:")
    jax.debug.visualize_array_sharding(
        params["params"]["model"]["layers_0"]["mlp"]["down_proj"]["kernel"].value
    )

    inputs = jax.device_put(
        inputs_jax["input_ids"], mesh_sharding(PartitionSpec("data", None))
    )
    print("Inputs:")
    jax.debug.visualize_array_sharding(inputs)
    outputs_jax = model_jax.apply(
        params,
        inputs,
    )

    print(outputs_jax[0].logits.shape)
    print("Sharding of output logits (batch, -1, vocab):")
    o = outputs_jax[0].logits[:, -1, :]
    jax.debug.visualize_array_sharding(o)

    # Assert that the output sharding is correct
    assert o.sharding.spec == PartitionSpec("data", "model")


def test_get_params():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    config = MistralConfig(
        hidden_size=128,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=4,
        sliding_window=3,
    )

    inputs_jax = tokenizer(
        ["Hello, my dog is cute", "Hello, my dog is cute"], return_tensors="jax"
    )

    model_jax = MistralForCausalLMJax(config)
    model = MistralForCausalLM(config)

    inputs = model_jax.prepare_input(inputs_jax["input_ids"])
    params = model_jax.get_params(
        weights=torch_to_jax_states(model, dtype=torch.float32, head_dim=config.hidden_size // config.num_attention_heads)
    )
    print(params)


def test_sharded_gen():
    """Test the simple api of sharded generation."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    config = MistralConfig(
        hidden_size=128,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=4,
        sliding_window=3,
    )

    inputs_jax = tokenizer(
        ["Hello, my dog is cute", "Hello, my dog is cute"], return_tensors="jax"
    )

    model_jax = MistralForCausalLMJax(config)

    inputs = model_jax.prepare_input(inputs_jax["input_ids"])
    params = model_jax.get_params()
    output = model_jax.generate(params, inputs, do_sample=True, max_length=10)


def test_save_load():
    config = MistralConfig(
        hidden_size=128,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=4,
        sliding_window=3,
    )
    model_jax = MistralForCausalLMJax(config)
    params = model_jax.get_params()

    abs_path = pathlib.Path(__file__).parent.absolute()
    if os.path.exists(str(abs_path) + "/tmp"):
        shutil.rmtree(str(abs_path) + "/tmp")

    save(params, str(abs_path) + "/tmp/")
    p = load(str(abs_path) + "/tmp/", item=params)
    print(params)
    print(p)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x, y: np.all(np.array(x) == np.array(y)), params, p))
    if os.path.exists(str(abs_path) + "/tmp"):
        shutil.rmtree(str(abs_path) + "/tmp")
