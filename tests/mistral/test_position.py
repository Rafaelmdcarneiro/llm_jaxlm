from jaxlm.nn.position import RotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import torch
import jax
import jax.numpy as jnp



def test_position(dim=12, max_length=2048, base=10000, max_log_len=10):
    with jax.default_device(jax.devices("cpu")[0]):
        emb_with_cache = RotaryEmbedding(dim=dim, max_length=max_length, base=base, disable_cache=False)
        emb_without_cache = RotaryEmbedding(dim=dim, max_length=max_length, base=base, disable_cache=True)

        ref = LlamaRotaryEmbedding(dim=dim, max_position_embeddings=max_length, base=base)

        key = jax.random.PRNGKey(0)
        q = jax.random.uniform(key, (8, 1, 16, dim), dtype=jnp.float32)
        params = emb_with_cache.init(key, q)
        params_1 = emb_without_cache.init(key, q)

        for l in map(lambda x: 2**x, range(2, max_log_len)):
            q = jax.random.uniform(key, (8, l, 16, dim), dtype=jnp.float32)
            q_t = torch.randn((8, 16, l, dim))
            pos = torch.arange(l).broadcast_to((8, l))

            cos, sin = emb_with_cache.apply(params, q)
            cos_1, sin_1 = emb_without_cache.apply(params_1, q)
            cos_ref, sin_ref = ref(q_t, pos)
            #print(cos, sin)
            #print(cos_1, sin_1)
            #print(cos_ref, sin_ref)
            assert jnp.allclose(cos, cos_1)
            assert jnp.allclose(sin, sin_1)
            print(cos.shape)
            print(cos_ref.shape)
            print((cos - cos_ref[0].numpy()).max())
            # note: huggingface llama rope broadcasts to batch axis (dumb!!)
            assert jnp.allclose(cos, cos_ref[0].numpy())
            assert jnp.allclose(sin, sin_ref[0].numpy())

