import functools

import jax
import jax.numpy as jnp

from .._generate import top_k_top_p_filtering


def generate(
    params,
    eval_fn,
    prompt_tokens: list | jnp.ndarray,
    do_sample: bool = True,
    seed: int = 0,
    max_len: int = 100,
    top_k: int = 0,
    top_p: float = 0.0,
    temp: float = 1.0,
):
    """
    **Debug only!** A naive implementation of decoding.
    Args:
        params: FrozenDict containing the model parameters
        eval_fn: the evaluation function (usually the `model.apply` or `jax.jit(model.apply)`)
        prompt_tokens: the tokenized prompt
        do_sample: whether to sample the distribution or take the argmax
        seed: random seed
        max_len: the max generation length
        top_k: top k
        top_p: top p
        temp: temperature
    Returns:
        the completed token array (containing the prompt)
    """
    if isinstance(prompt_tokens, list):
        current_state = jnp.array(prompt_tokens)
    elif len(prompt_tokens.shape) == 1:
        current_state = prompt_tokens[None, :]
    else:
        current_state = prompt_tokens

    past_key_values = None
    key = jax.random.PRNGKey(seed)
    for _ in range(max_len):
        if past_key_values is None:
            tok = current_state
        else:
            tok = current_state[:, -1:]
        outputs, past_key_values = eval_fn(
            params,
            tok,
            past_key_values=past_key_values,
            use_cache=True,
        )

        logits = outputs[:, -1:] * 1.0 / temp

        if do_sample:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            out_tk = jax.random.categorical(key, logits)
        else:
            out_tk = jnp.argmax(logits, axis=-1)

        current_state = jnp.concatenate((current_state, out_tk), axis=-1)
        key, subkey = jax.random.split(key)

    return current_state
