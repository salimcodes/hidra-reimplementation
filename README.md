# hidra-reimplementation

A from-scratch reimplementation of the core method below:

## The method

Activation steering adds a direction `d` to a transformer's residual-stream
activations to induce or suppress a behavior (e.g. `x <- x + alpha*d`, with
`d` estimated as a difference-in-means between contrastive activation sets).
HiDRA's claim is that this linear, first-order estimate misses discriminative
signal that only shows up in *nonlinear* combinations of activation
coordinates (e.g. per-class variance differences, motivated by the
superposition hypothesis). Instead of changing the steering algorithm, HiDRA
changes the space it operates in:

1. **Lift**: project the activation `x in R^d` into a much higher-dimensional
   space with a fixed random Gaussian matrix `A in R^(m x d)` (`m > d`)
   followed by an invertible elementwise nonlinearity `sigma`
   (LeakyReLU by default): `x' = sigma(A x)`.
2. **Steer**: estimate a difference-in-means vector `d` from contrastive
   prompt sets *in that lifted space*, and add it there: `x' + alpha*d`.
3. **Project back**: invert the lift to return to the residual stream:
   `x <- A_pinv( sigma^-1( sigma(A x) + alpha*d ) )` (paper's Eq. 9).

The paper proves (Prop. 3.1, Thm. 3.5) that this lifted-space Fisher
discriminant ratio is always >= the raw-space one, strictly greater whenever
a "residual" (non-linear) signal exists, and shows empirically that this
translates into stronger jailbreak/truthfulness/behavioral steering across
Gemma2, Llama3/3.2, and Qwen2.5 models.

## What's implemented here

The paper's experiments run on an H100 GPU across 9B-27B models, with
Llama-Guard-3 and fine-tuned GPT-4.1-nano judges. **This machine has no GPU**,
so rather than attempt (and fail to faithfully reproduce) those exact
numbers, this repo implements the method itself precisely and correctly, with
tests, and a small CPU-runnable demo that checks the mechanism actually
works and points in the right qualitative direction.

```
hidra/
  nonlinearities.py  # LeakyReLU, Cube, Cubic, Skip-softplus, Norm. skip-softplus (Sec 4.1, App D.1)
  projection.py      # RandomProjector: A, A_pinv, lift/unlift/steer (Eq. 9)
  steering.py        # difference-in-means (Eq. 1), Fisher ratio (Eq. 2) via a
                      # Woodbury-identity solve, LDA direction (App B.1)
  extraction.py       # residual-stream activation extraction via forward hooks
  pipeline.py         # HiDRASteerer: fit() contrastive vectors, generate() with steering
tests/                # correctness tests (no GPU/model download needed except a
                       # tiny randomly-initialized Llama used only to exercise generate())
examples/demo_qwen.py  # end-to-end demo on Qwen2.5-0.5B-Instruct (CPU)
```

### Tests (`pytest tests/`)

* Every nonlinearity actually inverts (`sigma^-1(sigma(t)) == t`), including
  the ones with no closed-form inverse (Cubic, Skip-softplus), which are
  inverted via Newton's method.
* The lift/unlift round trip exactly reconstructs `x` when `alpha=0` (since
  `m > d` gives `A` full column rank almost surely, so `A_pinv @ A = I`).
* Difference-in-means, the Fisher ratio, and the DiM/LDA equivalence under
  isotropic scatter (Lemma B.1) are checked against synthetic ground truth.
* **A direct empirical check of Proposition 3.1 / Theorem 3.5**: two
  synthetic classes with *identical means* but different per-coordinate
  variance (a purely second-order, superposition-style signal) give a
  near-zero raw-space Fisher ratio, while HiDRA's lifted-space Fisher ratio
  picks the signal up through the nonlinearity, exactly as the theory
  predicts.
* `HiDRASteerer.fit()`/`.generate()` are exercised end-to-end (hooks,
  prefill/decode step tracking, HiDRA vs. ActAdd vs. unsteered) against a
  tiny, randomly-initialized `LlamaForCausalLM` -- fully offline, no weights
  downloaded.

Run with:

```
pip install -r requirements.txt
pytest tests/ -q
```

### Demo (`examples/demo_qwen.py`)

Downloads `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0, ungated, ~1GB -- one of
the paper's three model families, at a size small enough for CPU), builds a
tiny (6 vs. 6) harmful/harmless contrastive prompt set in the spirit of the
paper's jailbreak setup (Sec. 5.1, which uses 128+128), and compares:

* no steering,
* **ActAdd**: DiM steering vector estimated and applied in the raw residual
  stream (the linear baseline the paper compares against),
* **HiDRA**: DiM steering vector estimated and applied in the lifted space
  (Eq. 9),

at the same `alpha`, on one held-out prompt, plus a Table-10-style Fisher
ratio diagnostic (original vs. lifted space).

```
PYTHONPATH=. python examples/demo_qwen.py --alpha 6 --m 2048 --max_new_tokens 50
```

This is a sanity check, not a benchmark: with 6 examples per class (vs. the
paper's 128) and no held-out evaluation set or ASR judge, it will not
reproduce Table 1's numbers. What it does show, reproducibly:

* the lifted-space Fisher ratio is orders of magnitude larger than the
  raw-space one (mirroring Table 10's diagnostic), and
* at equal `alpha`, HiDRA moves the generation further from the model's
  default refusal than ActAdd does.

At high `alpha` both methods degrade into the kind of incoherent,
"degenerate" generations the paper itself documents (Sec. 6.2, Appendix E)
when steering is pushed too hard for too little contrastive data -- so this
is also a real (if small-scale) illustration of a limitation the paper
discusses, not just of the method's benefit.

### CAA-style multiple-choice experiment (`examples/experiment_caa_mcqa.py`)

Unlike the demo above, this one *is* a real (if scaled-down) reproduction of
Table 3: for each behavioral concept, fit a DiM steering vector from
contrastive multiple-choice prompt pairs, then report the average next-token
probability assigned to the behavior-consistent answer letter on held-out
questions, under positive and negative steering, for no-steering vs. **CAA**
(raw-space DiM -- the paper's linear baseline) vs. **HiDRA** (lifted-space
DiM). It uses `HiDRASteerer.next_token_logits()`, a single forward pass per
question (no generation, no LLM judge needed), which is what makes this the
cheapest of the paper's three experiments to actually run on CPU.

Data: 5 of the paper's 6 concepts map onto Anthropic's public "Advanced AI
Risk" model-written evals and sycophancy datasets, matching Table 6's stated
sources exactly (`examples/caa_data.py`, auto-downloaded on first run). The
6th, **Hallucination**, was GPT-4-generated for the paper and isn't public;
`examples/data/hallucination_substitute.jsonl` is a hand-authored, 50-item
stand-in in the same schema (self-report questions about confabulating vs.
flagging uncertainty, following the framing of the paper's Table 7 system
prompts) -- clearly not the paper's dataset, just a substitute covering the
same idea. All 6 concepts (5 real + 1 substitute) are covered below.

```
PYTHONPATH=. python examples/experiment_caa_mcqa.py --n_train 40 --n_test 20
```

Model: `meta-llama/Llama-3.2-1B-Instruct` by default (gated -- requires
accepting the license on huggingface.co and an HF token), substituting for
the paper's Gemma-2-9B-IT since 9B doesn't fit comfortably in CPU RAM/time;
steering layer defaults to mid-depth (8 of 16) rather than the paper's tuned
layer 22. **Load the model in float32, not bfloat16** -- bfloat16 benchmarks
*~2x slower* on a CPU with no native bf16 support (it falls back to software
emulation), which is a real trap if you assume "lower precision = faster."

Actual result (40 train / 20 held-out test pairs per concept, `alpha in
{-2,-1}` / `{1,2}`, ~15 min total on one CPU core-bound run):

| Concept | No steer | CAA(-) | HiDRA(-) | CAA(+) | HiDRA(+) |
|---|---|---|---|---|---|
| ai_coordination | 0.387 | 0.185 | **0.183** | 0.637 | **0.640** |
| corrigibility | 0.668 | 0.256 | **0.234** | 0.754 | **0.758** |
| myopic_reward | 0.540 | 0.449 | **0.448** | **0.615** | 0.613 |
| survival_instinct | 0.350 | **0.234** | 0.238 | 0.409 | **0.419** |
| sycophancy | 0.558 | 0.471 | **0.468** | 0.651 | 0.651 (tie) |
| hallucination* | 0.242 | 0.098 | **0.090** | **0.556** | 0.555 |

\* substitute dataset, 30 train / 20 test (only 50 items available; run with
`--concepts hallucination --n_train 30 --n_test 20`).

(`-` column: lower = stronger suppression, better. `+` column: higher =
stronger elicitation, better. Bold = winner, matching the paper's own
convention in Table 3.)

HiDRA wins or ties CAA in 9/12 cells here -- the same qualitative direction
as Table 3 (HiDRA beats CAA on suppression in all 6 of the paper's concepts,
elicitation in 5/6) -- but with much thinner margins (e.g. the paper's
corrigibility gap is ~2x; ours is a few percent), consistent with a 1B model
and 30-40 training pairs standing in for a 9B model and hundreds.

## Known simplifications vs. the paper

* Table 3 (CAA-style MCQA) is reproduced at reduced scale (above); Tables
  1, 2, 4, 5 are not attempted (jailbreak/truthfulness need an LLM judge --
  Llama-Guard-3 or fine-tuned GPT judges -- and larger models than fit
  comfortably here; Table 5's nonlinearity ablation exists at the unit-test
  level, not as an ASR sweep).
* `LDA` extraction (Sec. 6.1) is implemented (`hidra.steering.lda_direction`)
  but not wired into either experiment script.
* SAS (the sparse-autoencoder steering baseline in Table 3) is not
  implemented -- it needs a pretrained SAE for the target model.
* Sequential multi-layer steering across *all* layers (the paper's default,
  following Rodriguez et al.) is supported (`HiDRASteerer(layers=[...])`
  accepts any subset), but both scripts here steer a single layer for speed.
