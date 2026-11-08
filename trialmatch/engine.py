"""Task 3: a from-scratch Qwen2 decoder forward pass with an explicit KV cache.

The engine reuses the (merged, fine-tuned) weights of a Hugging Face
Qwen2ForCausalLM but performs every step itself: embedding, RMSNorm, rotary
position embeddings (RoPE), grouped-query attention, SwiGLU MLP, final norm and
LM head. The KV cache is a plain list of tensors owned by this module, so cache
writes, positions, masking, growth and cropping (needed by speculative
decoding) are all explicit and inspectable.

Positional handling: a token at absolute position p is rotated by angle
p * inv_freq *before* its key is cached. Cached keys therefore never need to be
re-rotated; a new token only needs its own position (start_pos + i).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .attention import ATTENTION_FNS


class KVCache:
    """Pre-allocated per-layer K/V buffers [B, n_kv_heads, capacity, head_dim].

    `length` is the number of valid positions. Appending beyond capacity grows
    the buffers geometrically (x2) -- amortised O(1) per token instead of the
    O(T) re-allocation that naive torch.cat on every step would cost.
    """

    def __init__(self, n_layers, n_kv_heads, head_dim, dtype, device, batch=1, capacity=256):
        self.n_layers, self.shape = n_layers, (batch, n_kv_heads, capacity, head_dim)
        self.k = [torch.zeros(self.shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros(self.shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.length = 0
        self.capacity = capacity
        self.growth_events: list[int] = []

    def _ensure(self, needed: int):
        if needed <= self.capacity:
            return
        new_cap = self.capacity
        while new_cap < needed:
            new_cap *= 2
        for i in range(self.n_layers):
            for buf in (self.k, self.v):
                nb = torch.zeros((*buf[i].shape[:2], new_cap, buf[i].shape[-1]), dtype=buf[i].dtype, device=buf[i].device)
                nb[:, :, : self.length] = buf[i][:, :, : self.length]
                buf[i] = nb
        self.growth_events.append(new_cap)
        self.capacity = new_cap

    def write(self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor):
        """Write new keys/values at positions [start, start+T) and return the
        visible prefix [0, start+T) for attention."""
        end = start + k.shape[2]
        if layer == 0:
            self._ensure(end)
        self.k[layer][:, :, start:end] = k
        self.v[layer][:, :, start:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def crop(self, n: int):
        """Discard positions >= n (rejected speculative tokens). O(1): data past
        `length` is simply overwritten by later writes."""
        self.length = min(self.length, n)

    def snapshot(self) -> "KVCache":
        c = KVCache.__new__(KVCache)
        c.n_layers, c.shape, c.capacity, c.length = self.n_layers, self.shape, self.capacity, self.length
        c.k = [t.clone() for t in self.k]
        c.v = [t.clone() for t in self.v]
        c.growth_events = list(self.growth_events)
        return c

    def nbytes(self, valid_only: bool = True) -> int:
        n = self.length if valid_only else self.capacity
        per = self.k[0][:, :, :n].numel() * self.k[0].element_size()
        return 2 * self.n_layers * per


@dataclass
class EngineConfig:
    n_layers: int
    hidden: int
    n_heads: int
    n_kv: int
    head_dim: int
    eps: float
    rope_theta: float
    vocab: int


class QwenEngine:
    def __init__(self, hf_model, attention: str = "sdpa", block_q: int = 64, block_k: int = 64):
        cfg = hf_model.config
        assert cfg.model_type == "qwen2", "engine implements the Qwen2 architecture"
        rope = getattr(cfg, "rope_parameters", None) or {}
        theta = rope.get("rope_theta", getattr(cfg, "rope_theta", 10000.0))
        assert rope.get("rope_type", "default") == "default", "only default RoPE implemented"
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.cfg = EngineConfig(cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads,
                                cfg.num_key_value_heads, hd, cfg.rms_norm_eps, float(theta), cfg.vocab_size)
        self.m = hf_model.model
        self.lm_head = hf_model.lm_head
        self.device = next(hf_model.parameters()).device
        self.dtype = next(hf_model.parameters()).dtype
        self.set_attention(attention, block_q, block_k)
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32, device=self.device) / hd))
        self.eos_ids: set[int] = set()

    def set_attention(self, name: str, block_q: int = 64, block_k: int = 64):
        fn = ATTENTION_FNS[name]
        self.attention_name = name
        self.attn = (lambda q, k, v, q_offset: fn(q, k, v, causal=True, q_offset=q_offset, block_q=block_q, block_k=block_k)) \
            if name == "tiled" else (lambda q, k, v, q_offset: fn(q, k, v, causal=True, q_offset=q_offset))

    def new_cache(self, capacity: int = 256) -> KVCache:
        c = self.cfg
        return KVCache(c.n_layers, c.n_kv, c.head_dim, self.dtype, self.device, capacity=capacity)

    # ---------------------------------------------------------------- pieces
    def _rms(self, x, w):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.cfg.eps)
        return w * x.to(dt)

    def _rope(self, positions: torch.Tensor):
        freqs = positions.float()[:, None] * self.inv_freq[None, :]       # [T, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)                          # [T, D]
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    # --------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, ids: torch.Tensor, cache: KVCache | None = None, start_pos: int = 0,
                last_only: bool = False) -> torch.Tensor:
        """ids [1, T] occupying absolute positions [start_pos, start_pos+T).

        With a cache: new K/V are written at those positions and attention runs
        over the cached prefix plus the new tokens (bottom-right causal mask).
        Without a cache: start_pos must be 0 and the full sequence is processed.
        """
        c = self.cfg
        B, T = ids.shape
        if cache is not None:
            assert start_pos == cache.length, f"cache holds {cache.length} positions, writing at {start_pos}"
        else:
            assert start_pos == 0
        x = self.m.embed_tokens(ids)
        cos, sin = self._rope(torch.arange(start_pos, start_pos + T, device=self.device))
        groups = c.n_heads // c.n_kv
        for li, layer in enumerate(self.m.layers):
            at = layer.self_attn
            h = self._rms(x, layer.input_layernorm.weight)
            q = F.linear(h, at.q_proj.weight, at.q_proj.bias).view(B, T, c.n_heads, c.head_dim).transpose(1, 2)
            k = F.linear(h, at.k_proj.weight, at.k_proj.bias).view(B, T, c.n_kv, c.head_dim).transpose(1, 2)
            v = F.linear(h, at.v_proj.weight, at.v_proj.bias).view(B, T, c.n_kv, c.head_dim).transpose(1, 2)
            q = q * cos + self._rotate_half(q) * sin
            k = k * cos + self._rotate_half(k) * sin
            if cache is not None:
                k, v = cache.write(li, start_pos, k, v)      # K/V now cover [0, start_pos+T)
            if groups > 1:                                    # grouped-query attention
                k = k.repeat_interleave(groups, dim=1)
                v = v.repeat_interleave(groups, dim=1)
            o = self.attn(q, k, v, start_pos)
            o = o.transpose(1, 2).reshape(B, T, c.n_heads * c.head_dim)
            x = x + F.linear(o, at.o_proj.weight)
            mlp = layer.mlp
            h = self._rms(x, layer.post_attention_layernorm.weight)
            x = x + F.linear(F.silu(F.linear(h, mlp.gate_proj.weight)) * F.linear(h, mlp.up_proj.weight),
                             mlp.down_proj.weight)
        if cache is not None:
            cache.length = start_pos + T
        x = self._rms(x[:, -1:] if last_only else x, self.m.norm.weight)
        return self.lm_head(x).float()
