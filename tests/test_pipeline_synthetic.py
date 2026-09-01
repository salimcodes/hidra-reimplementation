"""End-to-end test of HiDRASteerer against a tiny, randomly-initialized
LlamaForCausalLM (no weight download -- fully offline), so the full
fit() -> generate() pipeline, including the layer hooks and the
prefill/decode step counter, is exercised without needing a real model.
"""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from hidra import HiDRASteerer


class _DummyBatch(dict):
    def to(self, device):
        return self


class DummyTokenizer:
    """Deterministic, offline stand-in for a real tokenizer."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, text, return_tensors="pt"):
        ids = [2 + (hash(w) % (self.vocab_size - 2)) for w in text.split()]
        if not ids:
            ids = [2]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return _DummyBatch(input_ids=input_ids, attention_mask=attention_mask)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def _make_tiny_model_and_tokenizer():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=63,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model, DummyTokenizer(vocab_size=64)


def test_fit_produces_correctly_shaped_vectors():
    model, tokenizer = _make_tiny_model_and_tokenizer()
    steerer = HiDRASteerer(model, tokenizer, layers=[0, 1], m=128, seed=0)
    steerer.fit(
        target_prompts=["alpha beta gamma", "delta epsilon"],
        source_prompts=["zeta eta theta", "iota kappa"],
    )
    for l in (0, 1):
        assert steerer.raw_vectors[l].shape == (32,)
        assert steerer.hidra_vectors[l].shape == (128,)


def test_fisher_ratio_report_structure():
    model, tokenizer = _make_tiny_model_and_tokenizer()
    steerer = HiDRASteerer(model, tokenizer, layers=[0], m=128, seed=0)
    steerer.fit(
        target_prompts=["alpha beta gamma", "delta epsilon", "mu nu xi"],
        source_prompts=["zeta eta theta", "iota kappa", "rho sigma tau"],
    )
    report = steerer.fisher_ratio_report(gammas=(1e-1,))
    assert 0 in report
    assert 1e-1 in report[0]
    assert set(report[0][1e-1]) == {"original", "hidra"}
    assert isinstance(report[0][1e-1]["original"], float)


def test_generate_runs_and_steering_changes_output():
    model, tokenizer = _make_tiny_model_and_tokenizer()
    steerer = HiDRASteerer(model, tokenizer, layers=[0], m=128, seed=0)
    steerer.fit(
        target_prompts=["alpha beta gamma", "delta epsilon", "mu nu xi"],
        source_prompts=["zeta eta theta", "iota kappa", "rho sigma tau"],
    )

    gen_kwargs = dict(max_new_tokens=6, do_sample=False)
    unsteered = steerer.generate("hello world", alpha=0.0, mode="none", **gen_kwargs)
    hidra_steered = steerer.generate("hello world", alpha=8.0, mode="hidra", **gen_kwargs)
    actadd_steered = steerer.generate("hello world", alpha=8.0, mode="actadd", **gen_kwargs)

    assert isinstance(unsteered, str) and isinstance(hidra_steered, str)
    # A strong steering strength on a tiny random model should perturb the
    # greedy decode relative to the unsteered baseline.
    assert hidra_steered != unsteered
    assert actadd_steered != unsteered


def test_prompt_only_scope_runs():
    model, tokenizer = _make_tiny_model_and_tokenizer()
    steerer = HiDRASteerer(model, tokenizer, layers=[0], m=128, seed=0)
    steerer.fit(
        target_prompts=["alpha beta gamma", "delta epsilon"],
        source_prompts=["zeta eta theta", "iota kappa"],
    )
    out = steerer.generate(
        "hello world", alpha=5.0, mode="hidra", scope="prompt_only", max_new_tokens=4, do_sample=False
    )
    assert isinstance(out, str)
