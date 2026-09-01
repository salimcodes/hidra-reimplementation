"""Residual-stream activation extraction from a HuggingFace causal LM.

Locates the per-layer decoder blocks (``model.model.layers``, the layout
shared by Llama/Gemma2/Qwen2-style architectures -- the model families used
in the paper) and captures the residual-stream hidden state at a chosen
token position via forward hooks, matching x^(l)_i(t) in the paper's
notation.
"""

from typing import Dict, List

import torch

Tensor = torch.Tensor


def _locate_decoder_layers(model) -> torch.nn.ModuleList:
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        return base.layers
    raise ValueError(
        "Could not locate decoder layers on this model (expected `model.model.layers`, "
        "the layout used by Llama/Gemma2/Qwen2-style architectures)."
    )


class ActivationExtractor:
    def __init__(self, model, tokenizer, layers: List[int]):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = list(layers)
        self.decoder_layers = _locate_decoder_layers(model)

    def extract(self, prompts: List[str], token_position: int = -1) -> Dict[int, Tensor]:
        """Run each prompt through the model and capture x^(l) at `token_position`
        (default: last token) for every requested layer. Returns
        {layer_idx: Tensor of shape (len(prompts), hidden_dim)}.
        """
        captured: Dict[int, List[Tensor]] = {l: [] for l in self.layers}

        def make_hook(layer_idx: int):
            def hook(module, inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                captured[layer_idx].append(hs[:, token_position, :].detach().to(torch.float32).cpu())

            return hook

        handles = [self.decoder_layers[l].register_forward_hook(make_hook(l)) for l in self.layers]
        device = next(self.model.parameters()).device
        try:
            self.model.eval()
            with torch.no_grad():
                for prompt in prompts:
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                    self.model(**inputs)
        finally:
            for h in handles:
                h.remove()

        return {l: torch.cat(captured[l], dim=0) for l in self.layers}
