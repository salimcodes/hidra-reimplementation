"""High-level HiDRA steering pipeline for a HuggingFace causal LM.

Wraps: contrastive activation extraction -> difference-in-means steering
vector (Eq. 1), estimated either in the raw residual stream (the ActAdd
baseline) or in HiDRA's lifted space (Sec. 4.2) -> steered generation via
Eq. 9, applied through forward hooks on the chosen decoder layer(s).
"""

from typing import Dict, List, Optional, Sequence

import torch

from .extraction import ActivationExtractor
from .projection import RandomProjector
from .steering import difference_in_means, fisher_ratio

Tensor = torch.Tensor


class HiDRASteerer:
    def __init__(
        self,
        model,
        tokenizer,
        layers: Sequence[int],
        m: int = 8192,
        nonlinearity: str = "leaky_relu",
        seed: int = 0,
        **nonlinearity_kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = list(layers)
        d = model.config.hidden_size
        self.projector = RandomProjector(d=d, m=m, nonlinearity=nonlinearity, seed=seed, **nonlinearity_kwargs)
        self.extractor = ActivationExtractor(model, tokenizer, self.layers)

        self.raw_vectors: Dict[int, Tensor] = {}
        self.hidra_vectors: Dict[int, Tensor] = {}
        self._fit_activations: Optional[tuple] = None

    def fit(self, target_prompts: Sequence[str], source_prompts: Sequence[str], token_position: int = -1) -> "HiDRASteerer":
        """Extract contrastive activations and compute both the raw-space DiM
        vector (ActAdd baseline) and the HiDRA (lifted-space) DiM vector, per layer.
        """
        target_acts = self.extractor.extract(list(target_prompts), token_position)
        source_acts = self.extractor.extract(list(source_prompts), token_position)
        self._fit_activations = (target_acts, source_acts)

        for l in self.layers:
            self.raw_vectors[l] = difference_in_means(target_acts[l], source_acts[l])
            lifted_target = self.projector.lift(target_acts[l])
            lifted_source = self.projector.lift(source_acts[l])
            self.hidra_vectors[l] = difference_in_means(lifted_target, lifted_source)
        return self

    def fisher_ratio_report(self, gammas: Sequence[float] = (1e-2, 1e-3, 1e-4)) -> Dict[int, Dict[float, Dict[str, float]]]:
        """Diagnostic mirroring Table 10: R(gamma) in the original activation
        space vs. in HiDRA's lifted space, for each fitted layer.
        """
        if self._fit_activations is None:
            raise RuntimeError("call fit() before fisher_ratio_report()")
        target_acts, source_acts = self._fit_activations
        report: Dict[int, Dict[float, Dict[str, float]]] = {}
        for l in self.layers:
            report[l] = {}
            lifted_target = self.projector.lift(target_acts[l])
            lifted_source = self.projector.lift(source_acts[l])
            for g in gammas:
                r_lin = fisher_ratio(target_acts[l], source_acts[l], g)
                r_phi = fisher_ratio(lifted_target, lifted_source, g)
                report[l][g] = {"original": r_lin, "hidra": r_phi}
        return report

    def generate(
        self,
        prompt: str,
        alpha: float = 0.0,
        mode: str = "hidra",
        scope: str = "all",
        **gen_kwargs,
    ) -> str:
        """Generate text with steering applied at the fitted layer(s).

        mode:  'hidra' (Eq. 9, steer in the lifted space), 'actadd' (plain
               x + alpha*d in the original space -- the linear baseline), or
               'none' (unsteered).
        scope: 'all' steers every forward pass (prefill and each decode
               step); 'prompt_only' steers only during the initial prefill,
               matching the paper's two reported settings.
        """
        if mode not in ("hidra", "actadd", "none"):
            raise ValueError(f"unknown mode '{mode}'")
        if scope not in ("all", "prompt_only"):
            raise ValueError(f"unknown scope '{scope}'")
        if mode != "none" and not self.hidra_vectors:
            raise RuntimeError("call fit() before generate()")

        state = {"step": -1}

        def bump_step(module, args):
            state["step"] += 1

        def make_hook(layer_idx: int):
            def hook(module, inputs, output):
                is_tuple = isinstance(output, tuple)
                hs = output[0] if is_tuple else output

                steer_now = mode != "none" and alpha != 0.0
                if steer_now and scope == "prompt_only":
                    steer_now = state["step"] == 0

                if steer_now:
                    orig_dtype = hs.dtype
                    x = hs.to(self.projector.dtype)
                    if mode == "hidra":
                        d_vec = self.hidra_vectors[layer_idx].to(x.device, x.dtype)
                        steered = self.projector.steer(x, d_vec, alpha)
                    else:  # actadd baseline
                        d_vec = self.raw_vectors[layer_idx].to(x.device, x.dtype)
                        steered = x + alpha * d_vec
                    hs = steered.to(orig_dtype)

                if is_tuple:
                    return (hs,) + output[1:]
                return hs

            return hook

        handles = [self.model.register_forward_pre_hook(bump_step)]
        handles += [self.extractor.decoder_layers[l].register_forward_hook(make_hook(l)) for l in self.layers]

        try:
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
            generated = output_ids[0, inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(generated, skip_special_tokens=True)
        finally:
            for h in handles:
                h.remove()
