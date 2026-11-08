"""Task 4: reference scaled dot-product attention and tiled online-softmax attention.

Shapes: q [B, H, Tq, D], k/v [B, H, Tk, D] (GQA heads already expanded).
`q_offset` is the absolute position of the first query. Query i may attend to
key j iff j <= q_offset + i. With a KV cache, Tk = q_offset + Tq, so the causal
mask is *bottom-right aligned*; during single-token decoding (Tq = 1) every
cached key is visible and no mask is needed at all.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


def causal_mask(tq: int, tk: int, q_offset: int, device) -> torch.Tensor:
    """Boolean [Tq, Tk] mask, True = allowed."""
    qpos = torch.arange(tq, device=device)[:, None] + q_offset
    kpos = torch.arange(tk, device=device)[None, :]
    return kpos <= qpos


def reference_attention(q, k, v, causal: bool = True, q_offset: int | None = None, scale: float | None = None):
    """softmax(Q K^T / sqrt(d) + M) V, materialising the full Tq x Tk score matrix.

    Numerically stable softmax: subtract the row max before exp (done in fp32).
    """
    tq, tk, d = q.shape[-2], k.shape[-2], q.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    if q_offset is None:
        q_offset = tk - tq
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale          # [B,H,Tq,Tk]
    if causal:
        s = s.masked_fill(~causal_mask(tq, tk, q_offset, q.device), NEG_INF)
    m = s.amax(dim=-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))                # fully-masked rows
    p = torch.exp(s - m)
    l = p.sum(dim=-1, keepdim=True)
    out = torch.matmul(p, v.float()) / l.clamp_min(1e-30)
    return out.to(q.dtype)


def tiled_attention(q, k, v, causal: bool = True, q_offset: int | None = None, scale: float | None = None,
                    block_q: int = 64, block_k: int = 64):
    """Blocked attention with online softmax (FlashAttention-style, pure PyTorch).

    For each query block we stream over key/value blocks keeping, per query row,
      m  : running maximum of the scores seen so far
      l  : running softmax normaliser  sum_j exp(s_j - m)
      acc: running un-normalised output sum_j exp(s_j - m) v_j
    When a new block raises the maximum to m_new, previous contributions are
    rescaled by alpha = exp(m_old - m_new). The full Tq x Tk matrix is never
    materialised: peak extra memory is O(block_q * block_k) per head.
    Key blocks lying entirely in the causal future of a query block are skipped.
    """
    B, H, tq, d = q.shape
    tk = k.shape[-2]
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    if q_offset is None:
        q_offset = tk - tq
    out = torch.empty((B, H, tq, d), dtype=torch.float32, device=q.device)
    kf, vf = k.float(), v.float()
    for qs in range(0, tq, block_q):
        qe = min(qs + block_q, tq)
        qb = q[:, :, qs:qe].float() * scale
        m = torch.full((B, H, qe - qs, 1), NEG_INF, device=q.device)
        l = torch.zeros((B, H, qe - qs, 1), device=q.device)
        acc = torch.zeros((B, H, qe - qs, d), device=q.device)
        last_q_pos = q_offset + qe - 1
        for ks in range(0, tk, block_k):
            if causal and ks > last_q_pos:
                break                                              # whole block is in the future
            ke = min(ks + block_k, tk)
            s = torch.matmul(qb, kf[:, :, ks:ke].transpose(-1, -2))   # [B,H,bq,bk]
            if causal and ke - 1 > q_offset + qs:                  # block straddles the diagonal
                mask = causal_mask(qe - qs, ke - ks, q_offset + qs - ks, q.device)
                s = s.masked_fill(~mask, NEG_INF)
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
            alpha = torch.exp(m - m_safe)                          # rescale old stats (exp(-inf)=0 at start)
            p = torch.exp(s - m_safe)
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            acc = alpha * acc + torch.matmul(p, vf[:, :, ks:ke])
            m = m_new
        out[:, :, qs:qe] = acc / l.clamp_min(1e-30)
    return out.to(q.dtype)


def sdpa_attention(q, k, v, causal: bool = True, q_offset: int | None = None, scale: float | None = None):
    """PyTorch fused SDPA. Note: is_causal=True assumes a TOP-LEFT aligned mask,
    which is wrong when Tq < Tk (cached decoding/verification), so an explicit
    bottom-right mask is passed in that case."""
    tq, tk = q.shape[-2], k.shape[-2]
    if q_offset is None:
        q_offset = tk - tq
    if not causal or tq == 1 and q_offset == tk - 1:
        return F.scaled_dot_product_attention(q, k, v, scale=scale)
    if tq == tk and q_offset == 0:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(tq, tk, q_offset, q.device), scale=scale)


ATTENTION_FNS = {"reference": reference_attention, "tiled": tiled_attention, "sdpa": sdpa_attention}
