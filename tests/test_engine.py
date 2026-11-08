"""Correctness of the own engine, KV-cache decoding and speculative decoding on
small random Qwen2 models (no downloads; fp32 CPU)."""
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from trialmatch.decoding import (engine_greedy, engine_no_cache_greedy, hf_cache_greedy, no_cache_greedy,
                                 speculative_greedy)
from trialmatch.engine import QwenEngine

VOCAB = 512


def tiny(layers, seed):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      tie_word_embeddings=True)
    return Qwen2ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def models():
    return tiny(3, 0), tiny(2, 1)


def prompt(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n,), generator=g).tolist()


@pytest.mark.parametrize("attn", ["sdpa", "reference", "tiled"])
def test_engine_logits_match_hf(models, attn):
    target, _ = models
    eng = QwenEngine(target, attention=attn, block_q=8, block_k=8)
    ids = torch.tensor([prompt(40)])
    hf = target(input_ids=ids).logits
    ours = eng.forward(ids)
    assert (hf - ours).abs().max() < 1e-4


@pytest.mark.parametrize("n_in,n_out", [(8, 16), (64, 32), (200, 48)])
def test_cache_paths_token_identical(models, n_in, n_out):
    target, _ = models
    eng = QwenEngine(target)
    p = prompt(n_in, n_in)
    ref, _ = no_cache_greedy(target, p, n_out)
    a, _ = hf_cache_greedy(target, p, n_out)
    b, st = engine_greedy(eng, p, n_out)
    c, _ = engine_no_cache_greedy(eng, p, n_out)
    assert ref == a == b == c
    assert st.extra["cache_len"] == n_in + n_out - 1   # last token is emitted, not yet cached


def test_cache_growth(models):
    eng = QwenEngine(models[0])
    cache = eng.new_cache(capacity=4)
    engine_greedy(eng, prompt(10), 20, cache=cache, prefix_len=0)
    assert cache.growth_events and cache.capacity >= 29


def test_prefix_cache_reuse(models):
    eng = QwenEngine(models[0])
    prefix, s1, s2 = prompt(30, 1), prompt(5, 2), prompt(9, 3)
    cache = eng.new_cache()
    eng.forward(torch.tensor([prefix]), cache, 0)
    out1, _ = engine_greedy(eng, prefix + s1, 10, cache=cache, prefix_len=len(prefix))
    out2, _ = engine_greedy(eng, prefix + s2, 10, cache=cache, prefix_len=len(prefix))
    assert out1 == engine_greedy(eng, prefix + s1, 10)[0]
    assert out2 == engine_greedy(eng, prefix + s2, 10)[0]


@pytest.mark.parametrize("k", [1, 2, 4, 7])
@pytest.mark.parametrize("n_in,n_out", [(10, 25), (50, 40)])
def test_speculative_equals_greedy(models, k, n_in, n_out):
    target, draft = models
    te, de = QwenEngine(target), QwenEngine(draft)
    p = prompt(n_in, 7)
    ref, _ = engine_greedy(te, p, n_out)
    out, st = speculative_greedy(te, de, p, n_out, k=k)
    assert out == ref
    assert 0.0 <= st.extra["acceptance_rate"] <= 1.0


def test_speculative_self_draft_accepts_all(models):
    target, _ = models
    te = QwenEngine(target)
    p = prompt(20, 9)
    out, st = speculative_greedy(te, QwenEngine(target), p, 30, k=4)
    assert out == engine_greedy(te, p, 30)[0]
    assert st.extra["acceptance_rate"] == 1.0


def test_speculative_eos(models):
    target, draft = models
    te, de = QwenEngine(target), QwenEngine(draft)
    p = prompt(15, 4)
    ref, _ = engine_greedy(te, p, 30)
    eos = {ref[10]}
    r2, _ = engine_greedy(te, p, 30, eos=eos)
    s2, _ = speculative_greedy(te, de, p, 30, k=3, eos=eos)
    assert r2 == s2 and r2[-1] in eos
