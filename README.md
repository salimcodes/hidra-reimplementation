# hidra-reimplementation

A from-scratch reimplementation of the core method from:

> Pham, Do, Abdullaev, Nguyen, Than. **"High-Dimensional Random Projection
> for Activation Steering in Language Models"** (HiDRA). arXiv:2606.15092v1
> (`2606.15092v1.pdf` in this repo).

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

## Known simplifications vs. the paper

* No exact reproduction of Tables 1-5 (model scale, GPU, judges, and dataset
  sizes are all out of reach on this machine).
* `LDA` extraction (Sec. 6.1) is implemented (`hidra.steering.lda_direction`)
  but not wired into the demo script.
* Sequential multi-layer steering across *all* layers (the paper's default,
  following Rodriguez et al.) is supported (`HiDRASteerer(layers=[...])`
  accepts any subset), but the demo only steers a single layer for speed.
