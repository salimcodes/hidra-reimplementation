"""High-level Angular Steering pipeline for a HuggingFace causal LM.

Mirrors hidra.pipeline.HiDRASteerer's structure and reuses its activation
extractor: fit() a behavior direction via difference-in-means over
contrastive prompts, then generate()/next_token_logits() with that direction
rotated into the residual stream at the fitted layer(s), by angle theta
(instead of hidra's alpha-scaled addition).
"""

from contextlib import contextmanager
from typing import Dict, Optional, Sequence

import torch

from hidra.extraction import ActivationExtractor
from hidra.steering import difference_in_means

from .rotation import rotate_toward_direction

Tensor = torch.Tensor


class AngularSteerer:
    def __init__(self, model, tokenizer, layers: Sequence[int]):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = list(layers)
        self.extractor = ActivationExtractor(model, tokenizer, self.layers)
        self.directions: Dict[int, Tensor] = {}  # unit d_hat per layer
        self._fit_activations: Optional[tuple] = None

    def fit(self, target_prompts: Sequence[str], source_prompts: Sequence[str], token_position: int = -1) -> "AngularSteerer":
        target_acts = self.extractor.extract(list(target_prompts), token_position)
        source_acts = self.extractor.extract(list(source_prompts), token_position)
        self._fit_activations = (target_acts, source_acts)
        for l in self.layers:
            d = difference_in_means(target_acts[l], source_acts[l])
            self.directions[l] = d / d.norm()
        return self

    @contextmanager
    def _rotation_hooks(self, theta: float, scope: str):
        if scope not in ("all", "prompt_only"):
            raise ValueError(f"unknown scope '{scope}'")
        if theta != 0.0 and not self.directions:
            raise RuntimeError("call fit() before steering")

        state = {"step": -1}

        def bump_step(module, args):
            state["step"] += 1

        def make_hook(layer_idx: int):
            def hook(module, inputs, output):
                is_tuple = isinstance(output, tuple)
                hs = output[0] if is_tuple else output

                rotate_now = theta != 0.0
                if rotate_now and scope == "prompt_only":
                    rotate_now = state["step"] == 0

                if rotate_now:
                    d_hat = self.directions[layer_idx].to(hs.device, hs.dtype)
                    hs = rotate_toward_direction(hs, d_hat, theta)

                if is_tuple:
                    return (hs,) + output[1:]
                return hs

            return hook

        handles = [self.model.register_forward_pre_hook(bump_step)]
        handles += [self.extractor.decoder_layers[l].register_forward_hook(make_hook(l)) for l in self.layers]
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def generate(self, prompt: str, theta: float = 0.0, scope: str = "all", **gen_kwargs) -> str:
        with self._rotation_hooks(theta, scope):
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
            generated = output_ids[0, inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(generated, skip_special_tokens=True)

    def next_token_logits(self, prompt: str, theta: float = 0.0) -> Tensor:
        with self._rotation_hooks(theta, scope="all"):
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = self.model(**inputs)
            return out.logits[0, -1, :].detach().cpu()
