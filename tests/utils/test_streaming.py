"""CPU unit tests for the streaming activation buffer.

The GPU choke point (`_trace_out`) is monkeypatched with a deterministic fake whose
activation value at every position equals the token id, so every buffered row is
attributable to the exact token that produced it. A minimal fake tokenizer stands in for
the HF one (the buffer only uses __call__, pad, bos_token and padding_side).
"""

import pytest
import torch

from diffing.utils.dictionary.streaming import (
    PairedActivationBuffer,
    _make_text_stream,
    _needs_special_tokens,
)

D_MODEL = 4
BOS_ID = 99


class FakeTokenizer:
    bos_token = "<bos>"
    padding_side = "left"

    def __call__(self, text, max_length=None, truncation=True, add_special_tokens=True):
        ids = [int(w) for w in text.split()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if add_special_tokens:
            ids = [BOS_ID] + ids
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def pad(self, encoded, padding=True, return_tensors="pt"):
        longest = max(len(e["input_ids"]) for e in encoded)
        ids, mask = [], []
        for e in encoded:
            n = longest - len(e["input_ids"])
            if self.padding_side == "right":
                ids.append(e["input_ids"] + [0] * n)
                mask.append(e["attention_mask"] + [0] * n)
            else:
                ids.append([0] * n + e["input_ids"])
                mask.append([0] * n + e["attention_mask"])
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


def fake_trace(dtype=torch.bfloat16, offset=0.0):
    """Activation of token t at every dim = t + offset (base offset 0, ft offset 1000)."""

    def _trace(self, model, submodule, tokens):
        ids = tokens["input_ids"].reshape(-1).to(torch.float32) + offset
        return ids.unsqueeze(-1).repeat(1, D_MODEL).to(dtype)

    return _trace


def make_buffer(monkeypatch, texts, dtype=torch.bfloat16, **kw):
    monkeypatch.setattr(PairedActivationBuffer, "_trace_out",
                        fake_trace(dtype), raising=True)
    defaults = dict(
        data=iter(texts), base_model=None, ft_model=None, base_submodule=None,
        ft_submodule=None, tokenizer=FakeTokenizer(), d_model=D_MODEL,
        context_len=8, refresh_batch_size=2, out_batch_size=4, n_ctxs=2,
        buffer_device="cpu", ignore_first_n_tokens=0, add_special_tokens=True,
    )
    defaults.update(kw)
    return PairedActivationBuffer(**defaults)


def buffered_token_values(buf):
    """The token ids reconstructed from the buffered activations (dim 0, base side)."""
    return sorted(int(v) for v in buf.activations[:, 0, 0].float().tolist())


def test_buffer_adopts_model_dtype_lazily(monkeypatch):
    buf = make_buffer(monkeypatch, ["1 2 3", "4 5 6"], dtype=torch.bfloat16)
    assert buf.activations.dtype == torch.float32          # empty scaffold
    buf.refresh()
    assert buf.activations.dtype == torch.bfloat16        # adopted at first fill
    assert buf.activations.shape[1:] == (2, D_MODEL)


def test_mask_token_and_ignore_first_are_dropped(monkeypatch):
    buf = make_buffer(monkeypatch, ["1 2 3", "4 2 5"], mask_token_id=2,
                      ignore_first_n_tokens=1)
    buf.refresh()
    vals = buffered_token_values(buf)
    assert 2 not in vals                                   # mask_token_id rows dropped
    assert BOS_ID not in vals                              # first token ignored
    assert set(vals) == {1, 3, 4, 5}
    # right-padding is forced when ignore_first_n_tokens > 0
    assert buf.tokenizer.padding_side == "right"


def test_padding_rows_never_buffered(monkeypatch):
    buf = make_buffer(monkeypatch, ["1 2 3 4 5", "6"])    # second text heavily padded
    buf.refresh()
    vals = buffered_token_values(buf)
    assert vals.count(0) == 0                              # attention-masked padding dropped
    assert {1, 2, 3, 4, 5, 6}.issubset(set(vals))


def test_next_yields_marks_read_and_exhausts(monkeypatch):
    buf = make_buffer(monkeypatch, ["1 2 3", "4 5 6"], out_batch_size=3)
    batch = next(buf)
    assert batch.shape == (3, 2, D_MODEL)
    assert int(buf.read.sum()) == 3
    seen = 3
    with pytest.raises(StopIteration):
        while True:
            seen += len(next(buf))
    assert seen == 8                                       # 2 texts x (3 tokens + BOS)


def test_base_and_ft_sides_kept_separate(monkeypatch):
    calls = []

    def side_trace(self, model, submodule, tokens):
        offset = 0.0 if len(calls) % 2 == 0 else 1000.0    # base traced first, then ft
        calls.append(offset)
        ids = tokens["input_ids"].reshape(-1).to(torch.float32) + offset
        return ids.unsqueeze(-1).repeat(1, D_MODEL)

    monkeypatch.setattr(PairedActivationBuffer, "_trace_out", side_trace, raising=True)
    buf = PairedActivationBuffer(
        data=iter(["1 2"]), base_model=None, ft_model=None, base_submodule=None,
        ft_submodule=None, tokenizer=FakeTokenizer(), d_model=D_MODEL,
        context_len=8, refresh_batch_size=2, out_batch_size=4, n_ctxs=2)
    buf.refresh()
    assert (buf.activations[:, 1, 0] - buf.activations[:, 0, 0] == 1000).all()


def test_normalizer_matches_manual_biased_std(monkeypatch):
    buf = make_buffer(monkeypatch, ["1 2 3", "4 5 6"], dtype=torch.float32)
    mean, std = buf.compute_normalizer(n_samples=10_000)
    sample = buf.activations.float()
    assert torch.allclose(mean, sample.mean(dim=0))
    assert torch.allclose(std, sample.std(dim=0, unbiased=False))
    assert mean.shape == (2, D_MODEL) and std.shape == (2, D_MODEL)


def test_needs_special_tokens_mirrors_disk_path():
    tok = FakeTokenizer()
    assert _needs_special_tokens("plain text", tok)
    assert not _needs_special_tokens("<bos> already templated", tok)


def test_text_stream_is_weighted_and_seeded():
    class DS(list):
        pass

    ds_a, ds_b = DS(["a"] * 50), DS(["b"] * 50)
    cfg = type("C", (), {"is_chat": False, "text_column": None})()

    import diffing.utils.dictionary.streaming as S
    orig = S._row_to_text
    S._row_to_text = lambda row, ds_cfg, tok: row
    try:
        stream = _make_text_stream([ds_a, ds_b], [cfg, cfg], FakeTokenizer(),
                                   draw_weights=[3.0, 1.0], seed=7)
        draws = [next(stream) for _ in range(2000)]
        frac_a = draws.count("a") / len(draws)
        assert 0.70 < frac_a < 0.80                        # ~3:1 weighting
        stream2 = _make_text_stream([ds_a, ds_b], [cfg, cfg], FakeTokenizer(),
                                    draw_weights=[3.0, 1.0], seed=7)
        assert [next(stream2) for _ in range(50)] == draws[:50]   # seeded determinism
    finally:
        S._row_to_text = orig
