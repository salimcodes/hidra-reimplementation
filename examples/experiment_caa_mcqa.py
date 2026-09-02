"""CAA-style multiple-choice steering experiment (paper Sec. 5.3 / Table 3).

Scaled down to run on CPU with a small model (default: google/gemma-2-2b-it,
the paper's Table 3 model family, at a size that fits on CPU) and a smaller
per-concept sample size than the paper's App. C.2.1 (Table 6), but otherwise
the same evaluation: for each behavioral concept, fit a difference-in-means
steering vector from contrastive prompt pairs, then report the average
next-token probability assigned to the behavior-consistent answer letter on
held-out questions, under positive and negative steering, for:

  * no steering
  * CAA   (DiM vector estimated & applied in the raw residual stream --
           this is exactly the paper's linear ActAdd/CAA baseline)
  * HiDRA (DiM vector estimated & applied in the lifted space, Eq. 9)

(SAS, the sparse-autoencoder baseline, needs a pretrained SAE for the target
model and is not implemented here.)
"""

import argparse
import re
import statistics
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from caa_data import CONCEPT_FILES, LOCAL_CONCEPT_FILES, load_concept

ALL_CONCEPTS = list(CONCEPT_FILES) + list(LOCAL_CONCEPT_FILES)
from hidra import HiDRASteerer

Tensor = torch.Tensor


def _normalize_question(question: str) -> str:
    """Some source files (e.g. survival-instinct.jsonl, the sycophancy
    files) already end their `question` field with "...Answer:", others
    don't. Strip it off if present so we can append our own "\\nAnswer:"
    consistently across every concept.
    """
    q = question.rstrip()
    if q.endswith("Answer:"):
        q = q[: -len("Answer:")].rstrip()
    return q


def make_prompts(item: dict) -> Tuple[str, str, str]:
    """Returns (positive_prompt, negative_prompt, eval_prompt).

    positive/negative prompts are used for DiM extraction (Sec 2.1, Eq 1):
    same question, differing only in the appended answer letter, matching
    the paper's App C.2.1 description. eval_prompt stops right before the
    letter, so the model's next token is scored directly.
    """
    base = f"{_normalize_question(item['question'])}\nAnswer:"
    positive = base + item["answer_matching_behavior"]
    negative = base + item["answer_not_matching_behavior"]
    eval_prompt = base + " ("
    return positive, negative, eval_prompt


def extract_letter(answer_str: str) -> str:
    m = re.search(r"\(([A-Z])\)", answer_str)
    if not m:
        raise ValueError(f"could not find a choice letter in '{answer_str}'")
    return m.group(1)


def choice_token_id(tokenizer, prefix: str, letter: str) -> int:
    """Token id for `letter` as it would actually be tokenized following
    `prefix`, found by diffing tokenize(prefix) against tokenize(prefix+letter)
    rather than guessing the tokenizer's internal vocab layout.
    """
    ids_prefix = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    ids_with = tokenizer(prefix + letter, add_special_tokens=False)["input_ids"]
    if len(ids_with) > len(ids_prefix) and ids_with[: len(ids_prefix)] == ids_prefix:
        return ids_with[len(ids_prefix)]
    # Fallback: tokenize the bare letter.
    fallback = tokenizer(letter, add_special_tokens=False)["input_ids"]
    return fallback[0]


def eval_concept(
    steerer: HiDRASteerer,
    test_items: List[dict],
    alpha: float,
    mode: str,
) -> float:
    """Average next-token probability assigned to the behavior-consistent
    answer letter over the held-out test items (paper's Sec. 5.3 metric).
    """
    probs = []
    for item in test_items:
        _, _, eval_prompt = make_prompts(item)
        matching_letter = extract_letter(item["answer_matching_behavior"])
        token_id = choice_token_id(steerer.tokenizer, eval_prompt, matching_letter)
        logits = steerer.next_token_logits(eval_prompt, alpha=alpha, mode=mode)
        prob = torch.softmax(logits, dim=-1)[token_id].item()
        probs.append(prob)
    return statistics.mean(probs)


def best_over_alphas(steerer, test_items, alphas: List[float], mode: str, want_max: bool) -> float:
    values = [eval_concept(steerer, test_items, a, mode) for a in alphas]
    return max(values) if want_max else min(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--layer", type=int, default=8, help="decoder layer to steer (paper uses layer 22 of Gemma-2-9B-IT; default here is mid-depth for a 16-layer 1B model)")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--slope", type=float, default=0.7, help="LeakyReLU slope (paper uses 0.7 in Table 3's setup)")
    parser.add_argument("--n_train", type=int, default=40, help="contrastive pairs used to fit the steering vector")
    parser.add_argument("--n_test", type=int, default=20, help="held-out questions evaluated per concept")
    parser.add_argument("--neg_alphas", type=float, nargs="+", default=[-2.0, -1.0])
    parser.add_argument("--pos_alphas", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--concepts", nargs="+", default=ALL_CONCEPTS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading {args.model} (CPU, float32)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # float32 benchmarked ~2x faster than bfloat16 on this CPU (no native
    # bf16 support, so bf16 falls back to slow software emulation).
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    num_layers = model.config.num_hidden_layers
    layer = min(args.layer, num_layers - 1)
    print(f"{num_layers} layers, hidden_size={model.config.hidden_size}; steering at layer {layer}\n", flush=True)

    header = f"{'Concept':<18}{'No steer':>10}{'CAA(-)':>10}{'HiDRA(-)':>10}{'CAA(+)':>10}{'HiDRA(+)':>10}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for concept in args.concepts:
        print(f"[{concept}] loading data + fitting steering vectors...", flush=True)
        train_items, test_items = load_concept(concept, args.n_train, args.n_test, seed=args.seed)

        steerer = HiDRASteerer(model, tokenizer, layers=[layer], m=args.m, nonlinearity="leaky_relu", seed=0, slope=args.slope)
        target_prompts, source_prompts = [], []
        for item in train_items:
            pos, neg, _ = make_prompts(item)
            target_prompts.append(pos)
            source_prompts.append(neg)
        steerer.fit(target_prompts, source_prompts)

        print(f"[{concept}] evaluating (no steering)...", flush=True)
        no_steer = eval_concept(steerer, test_items, alpha=0.0, mode="none")
        print(f"[{concept}] evaluating (negative alpha)...", flush=True)
        caa_neg = best_over_alphas(steerer, test_items, args.neg_alphas, mode="actadd", want_max=False)
        hidra_neg = best_over_alphas(steerer, test_items, args.neg_alphas, mode="hidra", want_max=False)
        print(f"[{concept}] evaluating (positive alpha)...", flush=True)
        caa_pos = best_over_alphas(steerer, test_items, args.pos_alphas, mode="actadd", want_max=True)
        hidra_pos = best_over_alphas(steerer, test_items, args.pos_alphas, mode="hidra", want_max=True)

        print(f"{concept:<18}{no_steer:>10.3f}{caa_neg:>10.3f}{hidra_neg:>10.3f}{caa_pos:>10.3f}{hidra_pos:>10.3f}", flush=True)


if __name__ == "__main__":
    main()
