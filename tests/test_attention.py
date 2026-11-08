import math

import pytest
import torch

from trialmatch.attention import reference_attention, sdpa_attention, tiled_attention


def naive_loop(q, k, v, q_offset):
    """Independent scalar-loop oracle (tiny shapes only)."""
    B, H, tq, d = q.shape
    out = torch.zeros_like(q, dtype=torch.float64)
    for b in range(B):
        for h in range(H):
            for i in range(tq):
                s = [(q[b, h, i].double() @ k[b, h, j].double()).item() / math.sqrt(d)
                     for j in range(q_offset + i + 1)]
                m = max(s)
                w = [math.exp(x - m) for x in s]
                z = sum(w)
                out[b, h, i] = sum(w[j] / z * v[b, h, j].double() for j in range(len(w)))
    return out


def test_reference_matches_scalar_oracle():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 5, 4) for _ in range(3))
    ref = reference_attention(q, k, v, causal=True)
    assert torch.allclose(ref.double(), naive_loop(q, k, v, 0), atol=1e-5)


@pytest.mark.parametrize("B,H,tq,tk,d", [(1, 1, 7, 7, 8), (2, 4, 64, 64, 32), (1, 2, 130, 130, 64),
                                         (2, 3, 1, 50, 16), (1, 2, 5, 37, 16)])
@pytest.mark.parametrize("bq,bk", [(16, 16), (32, 8), (64, 64)])
def test_tiled_matches_reference(B, H, tq, tk, d, bq, bk):
    torch.manual_seed(1)
    q = torch.randn(B, H, tq, d); k = torch.randn(B, H, tk, d); v = torch.randn(B, H, tk, d)
    for causal in [True, False]:
        ref = reference_attention(q, k, v, causal=causal)
        til = tiled_attention(q, k, v, causal=causal, block_q=bq, block_k=bk)
        assert (ref - til).abs().max() < 1e-5


def test_large_logits_stable():
    torch.manual_seed(2)
    q = torch.randn(1, 1, 20, 8) * 100   # scores ~1e4: naive exp would overflow
    k = torch.randn(1, 1, 20, 8) * 100
    v = torch.randn(1, 1, 20, 8)
    til = tiled_attention(q, k, v, block_q=4, block_k=4)
    ref = reference_attention(q, k, v)
    # scores ~1e4 have an fp32 spacing of ~1e-3, so summation order alone causes ~1e-4 differences
    assert torch.isfinite(til).all() and (ref - til).abs().max() < 1e-3


def test_sdpa_bottom_right_mask():
    torch.manual_seed(3)
    q = torch.randn(1, 2, 4, 16); k = torch.randn(1, 2, 10, 16); v = torch.randn(1, 2, 10, 16)
    assert (sdpa_attention(q, k, v, q_offset=6) - reference_attention(q, k, v, q_offset=6)).abs().max() < 1e-5
