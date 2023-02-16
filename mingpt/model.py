"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""


import math
from functools import partial

from jax import lax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.linear import DenseGeneral
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core import freeze, unfreeze
import numpy as np

from mingpt.utils import CfgNode as CN

from typing import Any
Array = Any
# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * jnp.power(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """
    config: Any
    causal: bool = False    

    def setup(self):
        self.c_attn = nn.Dense(3 * self.config.n_embd, kernel_init=self.config.kernel_init,
                               bias_init=self.config.bias_init)
        self.c_proj = nn.Dense(self.config.n_embd, kernel_init=self.config.kernel_init,
                               bias_init=self.config.bias_init)
        causal_mask = nn.make_causal_mask(jnp.ones((1, self.config.block_size),dtype="bool"), dtype="bool")
        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd

    @nn.compact
    def _kv_cache(self, key, value, query, attention_mask):
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, num_heads, max_length, depth_per_head = cached_key.value.shape
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (0, cur_index, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[2]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = jnp.logical_and(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(self, x: Array, init_cache:bool, training:bool):
        assert self.config.n_embd % self.config.n_head == 0


        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)
        dtype = x.dtype

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = jnp.split(self.c_attn(x), 3, axis=2)
        k = k.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2) # (B, nh, T, hs)
        q = q.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2) # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_head, C // self.n_head).swapaxes(1, 2) # (B, nh, T, hs)

        query_length, key_length = q.shape[2], k.shape[2]
        if self.causal:
            if self.has_variable("cache", "cached_key"):
                mask_shift = self.variables["cache"]["cache_index"]
                max_decoder_length = self.variables["cache"]["cached_key"].shape[2]
                causal_mask = lax.dynamic_slice(
                    self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
                )
            else:
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]
            causal_mask = jnp.broadcast_to(causal_mask, (B,) + causal_mask.shape[1:])

        if self.causal and (self.has_variable("cache", "cached_key") or init_cache):
            k, v, mask = self._kv_cache(k, v, q, causal_mask)
        
        attention_bias = lax.select(
            mask > 0,
            jnp.full(mask.shape, 0.0).astype(dtype),
            jnp.full(mask.shape, jnp.finfo(dtype).min).astype(dtype,
        ))

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.swapaxes(-2, -1)) * (1.0 / jnp.sqrt(k.shape[-1]))

        att = att + attention_bias  # apply mask
        att = nn.softmax(att, axis=-1)
        attn_dropout = nn.Dropout(rate=self.config.attn_pdrop)
        att = attn_dropout(att, deterministic=not training)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.swapaxes(1, 2).reshape(B,T,C) # re-assemble all head outputs side by side

        # output projection
        resid_dropout = nn.Dropout(rate=self.config.resid_pdrop)
        y = resid_dropout(self.c_proj(y), deterministic=not training)
        return y


class MLP(nn.Module):
    """An unassuming MLP block."""
    config: Any

    @nn.compact
    def __call__(self, x: Array, training: bool):
        x = nn.Dense(4 * self.config.n_embd, name='c_fc')(x)
        x = NewGELU(x)
        x = nn.Dense(self.config.n_embd, name='c_proj')(x)
        return nn.Dropout(rate=self.config.resid_pdrop)(x, deterministic=not training)


class Block(nn.Module):
    """an unassuming transformer block"""
    config: Any

    def setup(self):
        self.ln_1 = nn.LayerNorm()
        self.attn = CausalSelfAttention(self.config)
        self.ln_2 = nn.LayerNorm()
        self.mlp = MLP(self.config)


    def __call__(self, x, mask: Array, training: bool):
        x = x + self.attn(self.ln_1(x), mask, training)
        x = x + self.mlp(self.ln_2(x), training)
        return x


class Layers(nn.Module):
    """Heads"""
    config: Any

    @nn.compact
    def __call__(self, x: Array, init_cache: bool, training: bool):
        for i in range(self.config.n_layer):
            x = Block(self.config, name=str(i))(x, init_cache, training)
        return x


class GPT(nn.Module):
    """ GPT Language Model """
    config: Any
    
    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def setup(self):
        self.block_size = self.config.block_size
        self.wte = nn.Embed(self.config.vocab_size, self.config.n_embd)
        self.wpe = nn.Embed(self.config.block_size, self.config.n_embd)
        self.drop = nn.Dropout(rate=self.config.embd_pdrop)
        self.h = Layers(self.config, name='h')
        self.ln_f = nn.LayerNorm()
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False)

    @classmethod
    def get_specifications(cls, config):
        assert config.vocab_size is not None
        assert config.block_size is not None

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)

        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])
        
        return config

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import FlaxGPT2LMHeadModel
        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)


        # init a huggingface/transformers model
        model_hf = FlaxGPT2LMHeadModel.from_pretrained(model_type)
        hf_params = model_hf.params['transformer']
        hf_params = flatten_dict(hf_params, sep='.')
        transposed = ['attn.c_attn.kernel', 'attn.c_proj.kernel', 'mlp.c_fc.kernel', 'mlp.c_proj.kernel']

        for k in hf_params.keys():
            if any(k.endswith(w) for w in transposed):
                hf_params[k] = hf_params[k].T
        
        hf_params = unflatten_dict(hf_params, sep='.')

        return model, freeze(hf_params)

    def __call__(self, idx, training: bool):
        b, t = idx.shape
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        
        pos = jnp.arange(0, t, dtype=jnp.int64)[None] # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb, deterministic=not training)

        # causal mask to ensure that attention is only applied to the left in the input sequence

        x = self.h(x, init_cache, training)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.shape[1] <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                dtype = logits.dtype
                big_neg = jnp.finfo(dtype).min
                v, _ = lax.top_k(logits, top_k)
                logits = jnp.where(logits < v[:, [-1]], big_neg, logits)
            # apply softmax to convert logits to (normalized) probabilities
            probs = nn.activation.softmax(logits, axis=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                rvs = np.random.multinomial(1, probs)
                idx_next = rvs.argmax(axis=-1, keepdims=True)
            else:
                _, idx_next = lax.top_k(probs, k=1)
            # append sampled index to the running sequence and continue
            idx = jnp.concatenate((idx, idx_next), axis=1)