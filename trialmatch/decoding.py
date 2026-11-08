"""Greedy decoding paths for Tasks 3 and 5, all with the same timing protocol.

  no_cache_greedy      reference: re-runs the full sequence every step (O(T^2) total)
  hf_cache_greedy      HF model forward with an explicitly managed DynamicCache,
                       explicit position_ids / cache_position / attention_mask
  engine_greedy        own Qwen2 forward + own KVCache (engine.py)
  speculative_greedy   draft/target greedy speculative decoding on the engine

Every function returns (generated_token_ids, stats). Timing uses
torch.cuda.synchronize() around each phase on GPU so asynchronous kernels are
counted in the phase that launched them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .engine import KVCache, QwenEngine


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _mem_reset(device):
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def _mem_peak(device):
    return torch.cuda.max_memory_allocated() / 2**20 if str(device).startswith("cuda") else None


@dataclass
class GenStats:
    prefill_s: float = 0.0
    decode_s: float = 0.0
    n_prompt: int = 0
    n_new: int = 0
    peak_mem_mb: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def total_s(self):
        return self.prefill_s + self.decode_s

    @property
    def decode_tok_s(self):
        # tokens produced after the first one, per second of decode phase
        return (self.n_new - 1) / self.decode_s if self.decode_s > 0 and self.n_new > 1 else 0.0

    @property
    def e2e_tok_s(self):
        return self.n_new / self.total_s if self.total_s > 0 else 0.0

    def as_dict(self):
        return {"prefill_s": self.prefill_s, "decode_s": self.decode_s, "total_s": self.total_s,
                "n_prompt": self.n_prompt, "n_new": self.n_new, "decode_tok_s": self.decode_tok_s,
                "e2e_tok_s": self.e2e_tok_s, "peak_mem_mb": self.peak_mem_mb, **self.extra}


def _stop(tok: int, eos: set[int] | None) -> bool:
    return eos is not None and tok in eos


# ------------------------------------------------------------- reference path
@torch.no_grad()
def no_cache_greedy(model, prompt_ids: list[int], max_new: int, eos: set[int] | None = None):
    """Reference without cache reuse: every step recomputes all K/V from scratch."""
    dev = next(model.parameters()).device
    _mem_reset(dev); _sync(dev)
    ids = torch.tensor([prompt_ids], device=dev)
    out: list[int] = []
    st = GenStats(n_prompt=len(prompt_ids))
    t0 = time.perf_counter()
    for step in range(max_new):
        logits = model(input_ids=ids, use_cache=False).logits[:, -1]
        nxt = int(logits.argmax(-1))
        if step == 0:
            _sync(dev); st.prefill_s = time.perf_counter() - t0; t1 = time.perf_counter()
        out.append(nxt)
        if _stop(nxt, eos):
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
    _sync(dev)
    st.decode_s = time.perf_counter() - t1
    st.n_new, st.peak_mem_mb = len(out), _mem_peak(dev)
    return out, st


# ------------------------------------------ HF forward + explicit DynamicCache
@torch.no_grad()
def hf_cache_greedy(model, prompt_ids: list[int], max_new: int, eos: set[int] | None = None):
    """Cache-aware loop on the HF module: we create the cache, pass it in, and
    supply positions ourselves. The prompt is processed once (prefill); each
    step then feeds ONLY the newest token with position_ids = current length."""
    from transformers import DynamicCache

    dev = next(model.parameters()).device
    _mem_reset(dev); _sync(dev)
    cache = DynamicCache(config=model.config)
    T = len(prompt_ids)
    st = GenStats(n_prompt=T)
    t0 = time.perf_counter()
    ids = torch.tensor([prompt_ids], device=dev)
    o = model(input_ids=ids, past_key_values=cache, use_cache=True,
              position_ids=torch.arange(T, device=dev)[None], cache_position=torch.arange(T, device=dev),
              attention_mask=torch.ones((1, T), dtype=torch.long, device=dev))
    nxt = int(o.logits[:, -1].argmax(-1))
    _sync(dev); st.prefill_s = time.perf_counter() - t0
    out = [nxt]
    t1 = time.perf_counter()
    while len(out) < max_new and not _stop(nxt, eos):
        pos = cache.get_seq_length()          # == T + len(out) - 1
        o = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=cache, use_cache=True,
                  position_ids=torch.tensor([[pos]], device=dev), cache_position=torch.tensor([pos], device=dev),
                  attention_mask=torch.ones((1, pos + 1), dtype=torch.long, device=dev))
        nxt = int(o.logits[:, -1].argmax(-1))
        out.append(nxt)
    _sync(dev)
    st.decode_s = time.perf_counter() - t1
    st.n_new, st.peak_mem_mb = len(out), _mem_peak(dev)
    st.extra["cache_len"] = cache.get_seq_length()
    return out, st


# ------------------------------------------------------- own engine + KVCache
@torch.no_grad()
def engine_greedy(engine: QwenEngine, prompt_ids: list[int], max_new: int, eos: set[int] | None = None,
                  cache: KVCache | None = None, prefix_len: int = 0):
    """Own-engine greedy decoding. If `cache` already holds the first
    `prefix_len` prompt tokens (prefix caching across criteria), only the
    remaining prompt tokens are prefilled."""
    dev = engine.device
    _mem_reset(dev); _sync(dev)
    if cache is None:
        cache, prefix_len = engine.new_cache(capacity=max(256, len(prompt_ids) + max_new)), 0
    else:
        cache.crop(prefix_len)
    st = GenStats(n_prompt=len(prompt_ids))
    t0 = time.perf_counter()
    rest = torch.tensor([prompt_ids[prefix_len:]], device=dev)
    logits = engine.forward(rest, cache, start_pos=prefix_len, last_only=True)
    nxt = int(logits[0, -1].argmax())
    _sync(dev); st.prefill_s = time.perf_counter() - t0
    out = [nxt]
    t1 = time.perf_counter()
    while len(out) < max_new and not _stop(nxt, eos):
        logits = engine.forward(torch.tensor([[nxt]], device=dev), cache, start_pos=cache.length, last_only=True)
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
    _sync(dev)
    st.decode_s = time.perf_counter() - t1
    st.n_new, st.peak_mem_mb = len(out), _mem_peak(dev)
    st.extra.update(cache_len=cache.length, cache_mb=cache.nbytes() / 2**20, cache_growth=cache.growth_events,
                    prefix_reused=prefix_len)
    return out, st


@torch.no_grad()
def engine_no_cache_greedy(engine: QwenEngine, prompt_ids: list[int], max_new: int, eos: set[int] | None = None):
    """Uncached reference on the SAME engine code path (isolates the cache effect)."""
    dev = engine.device
    _mem_reset(dev); _sync(dev)
    seq = list(prompt_ids)
    out: list[int] = []
    st = GenStats(n_prompt=len(prompt_ids))
    t0 = time.perf_counter()
    for step in range(max_new):
        logits = engine.forward(torch.tensor([seq], device=dev), None, 0, last_only=True)
        nxt = int(logits[0, -1].argmax())
        if step == 0:
            _sync(dev); st.prefill_s = time.perf_counter() - t0; t1 = time.perf_counter()
        out.append(nxt); seq.append(nxt)
        if _stop(nxt, eos):
            break
    _sync(dev)
    st.decode_s = time.perf_counter() - t1
    st.n_new, st.peak_mem_mb = len(out), _mem_peak(dev)
    return out, st


# --------------------------------------------------- speculative decoding
@torch.no_grad()
def speculative_greedy(target: QwenEngine, draft: QwenEngine, prompt_ids: list[int], max_new: int, k: int = 4,
                       eos: set[int] | None = None, t_cache: KVCache | None = None, d_cache: KVCache | None = None,
                       prefix_len: int = 0):
    """Greedy speculative decoding (Leviathan et al. 2023; Chen et al. 2023).

    Invariant at the start of every round, with committed sequence `seq`:
      target cache holds K/V for seq[:-1]   (the last token is not yet processed)
      draft  cache holds K/V for a prefix of seq[:-1] (it may lag by one token)
    Round:
      1. draft proposes d_1..d_k greedily (feeding whatever it has not cached yet)
      2. target scores [seq[-1], d_1..d_k] in ONE forward pass -> t_1..t_{k+1}
         where t_i = argmax p_target(. | seq, d_1..d_{i-1})
      3. accept the longest prefix with d_i == t_i (n tokens), then append t_{n+1}
         (the correction on rejection, or a free bonus token if all k matched)
      4. crop both caches back to len(new_seq) - 1 so rejected K/V are discarded
    Under greedy decoding the output is token-identical to target-only greedy.
    """
    dev = target.device
    _mem_reset(dev); _sync(dev)
    cap = max(256, len(prompt_ids) + max_new + k + 2)
    if t_cache is None:
        t_cache, d_cache, prefix_len = target.new_cache(cap), draft.new_cache(cap), 0
    else:
        t_cache.crop(prefix_len); d_cache.crop(prefix_len)
    st = GenStats(n_prompt=len(prompt_ids))
    rounds, accepted_hist, drafted, accepted = 0, [], 0, 0
    t_draft = t_verify = 0.0

    # prefill both models on prompt[:-1] so the invariant holds; the last
    # prompt token is fed as the first token of round 1.
    t0 = time.perf_counter()
    if len(prompt_ids) - 1 > prefix_len:
        body = torch.tensor([prompt_ids[prefix_len:-1]], device=dev)
        target.forward(body, t_cache, prefix_len, last_only=True)
        draft.forward(body, d_cache, prefix_len, last_only=True)
    _sync(dev); st.prefill_s = time.perf_counter() - t0

    seq = list(prompt_ids)
    out: list[int] = []
    t1 = time.perf_counter()
    while len(out) < max_new:
        rounds += 1
        # --- 1. draft k tokens
        ta = time.perf_counter()
        kk = min(k, max_new - len(out))           # never draft past the budget
        props: list[int] = []
        feed = seq[d_cache.length:]               # tokens the draft has not cached (1 or 2)
        for _ in range(kk):
            lg = draft.forward(torch.tensor([feed], device=dev), d_cache, d_cache.length, last_only=True)
            nxt = int(lg[0, -1].argmax())
            props.append(nxt)
            feed = [nxt]
            if _stop(nxt, eos):
                break
        _sync(dev); t_draft += time.perf_counter() - ta
        # --- 2. verify with one target pass over [last committed token, drafts]
        tb = time.perf_counter()
        start = t_cache.length                    # == len(seq) - 1
        lg = target.forward(torch.tensor([[seq[-1]] + props], device=dev), t_cache, start)
        tgt = lg[0].argmax(-1).tolist()           # len(props)+1 target predictions
        _sync(dev); t_verify += time.perf_counter() - tb
        # --- 3. accept matching prefix + one target token
        n = 0
        while n < len(props) and props[n] == tgt[n]:
            n += 1
        new = props[:n] + [tgt[n]]
        drafted += len(props); accepted += n; accepted_hist.append(n)
        # respect eos / budget
        emitted = []
        for tkn in new:
            emitted.append(tkn)
            if _stop(tkn, eos) or len(out) + len(emitted) >= max_new:
                break
        seq += emitted; out += emitted
        # --- 4. cache synchronisation: keep K/V only for committed seq[:-1]
        t_cache.crop(len(seq) - 1)
        d_cache.crop(min(d_cache.length, len(seq) - 1))
        if _stop(out[-1], eos):
            break
    _sync(dev)
    st.decode_s = time.perf_counter() - t1
    st.n_new, st.peak_mem_mb = len(out), _mem_peak(dev)
    st.extra.update(k=k, rounds=rounds, drafted=drafted, accepted=accepted,
                    acceptance_rate=accepted / drafted if drafted else 0.0,
                    mean_accepted_per_round=accepted / rounds if rounds else 0.0,
                    tokens_per_target_call=len(out) / rounds if rounds else 0.0,
                    draft_s=t_draft, verify_s=t_verify, accept_hist=accepted_hist, prefix_reused=prefix_len)
    return out, st
