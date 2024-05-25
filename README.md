# jaxlm

A study playground for LM components in JAX. Started off as mistral-jax implementation but plan to cover a few other major LLMs as well.




# Old README:


## mistral_jax
(unofficial) Mistral model in JAX using TPU pod. Currently it is working on my GCP TPU v3-8 pods.

This repo is deeply indebted to the [Google TRC program](https://sites.research.google/trc/faq/) without which
none of the codes here could have been implemented, tested and used in research use-cases.

## Quickstart

```bash
pip install -e .
```
In Python:
```python
import jax
# Still works with HuggingFace's tokenizer and config
from transformers import AutoTokenizer, MistralForCausalLM

from mistral_jax import MistralForCausalLM as MistralForCausalLMJAX
from mistral_jax.utils import torch_to_jax_states
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# Tokenize the prompt
inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

# Initialize the JAX model
model_jax = MistralForCausalLMJAX(model.config)

# Get the initial parameters (esp. for the mutable variables)
key = jax.random.PRNGKey(0)
# Designate a mesh layout
mesh = (1, None)
# The input is sharded according to (1, ...) with axis names ("data", "model")
inputs = model_jax.prepare_input(inputs["input_ids"], device_mesh_layout=mesh)
# Converted state dict from the PyTorch model, and possibly shard the params
params = model_jax.get_params(weights=torch_to_jax_states(model.state_dict()), 
                              device_mesh_layout=mesh)
```
To obtain individual logit outputs and kv-cache:
```python
outputs, mutable_vars = model_jax.apply(
    params,
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    mutable=("cache",),
    output_hidden_states=True,
)
```
To perform a completion:
```python
out_jax = model_jax.generate(
    params, 
    inputs_jax["input_ids"], 
    do_sample=True, 
    max_length=10
)
completion = tokenizer.batch_decode(out_jax)
```

(ps: is there a way to let JAX compile different shapes slightly ahead of jit and in parallel?)

To save the checkpoint:
```python
from pathlib import Path
from mistral_jax.utils import save
abs_path = pathlib.Path(__file__).parent.absolute()
save(params, str(abs_path) + "/ckpt")
```
To load the checkpoint
```python
from mistral_jax.utils import load
p = load(str(abs_path) + "/ckpt")
```
