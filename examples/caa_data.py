"""Loader for the CAA-style multiple-choice behavior datasets used in the
paper's Sec. 5.3 / Table 3 (via Anthropic's public "Advanced AI Risk"
model-written evals, matching Table 6's stated sources).

Each item has the format used by both the original CAA paper (Rimsky et al.)
and this paper's App. C.2.1:

    {"question": "...\\n\\nChoices:\\n (A) ...\\n (B) ...",
     "answer_matching_behavior": " (A)",
     "answer_not_matching_behavior": " (B)"}

Five of the paper's six concepts map directly onto public files; the sixth,
Hallucination, was GPT-4-generated for the paper and isn't public, so it's
omitted here (see README).
"""

import json
import os
import random
from typing import Dict, List, Tuple

import requests

RAW_BASE = "https://raw.githubusercontent.com/anthropics/evals/main/"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "anthropic_evals")

# Matches Table 6's "Source / construction" column exactly, for the 5
# concepts with public source data.
CONCEPT_FILES: Dict[str, List[str]] = {
    "ai_coordination": ["advanced-ai-risk/human_generated_evals/coordinate-other-ais.jsonl"],
    "corrigibility": ["advanced-ai-risk/human_generated_evals/corrigible-neutral-HHH.jsonl"],
    "myopic_reward": ["advanced-ai-risk/human_generated_evals/myopic-reward.jsonl"],
    "survival_instinct": ["advanced-ai-risk/human_generated_evals/survival-instinct.jsonl"],
    "sycophancy": [
        "sycophancy/sycophancy_on_nlp_survey.jsonl",
        "sycophancy/sycophancy_on_political_typology_quiz.jsonl",
    ],
}

# The paper's 6th concept, Hallucination, was GPT-4-generated and isn't
# public. This is a hand-authored substitute in the same schema (50 items,
# self-report questions about confabulation vs. flagging uncertainty,
# following the framing of the paper's Table 7 system prompts) -- NOT the
# paper's actual dataset, just a stand-in covering the same idea.
LOCAL_CONCEPT_FILES: Dict[str, str] = {
    "hallucination": "hallucination_substitute.jsonl",
}

# Cap how many lines we keep from each source file on disk -- we only ever
# need a couple hundred items per concept, and some source files (sycophancy)
# have ~10k lines.
MAX_LINES_PER_FILE = 500


def _ensure_downloaded(rel_path: str) -> str:
    local_path = os.path.join(DATA_DIR, rel_path.replace("/", os.sep))
    if os.path.exists(local_path):
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    url = RAW_BASE + rel_path
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    lines = resp.text.splitlines()[:MAX_LINES_PER_FILE]
    with open(local_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return local_path


def _load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_concept(name: str, n_train: int, n_test: int, seed: int = 0) -> Tuple[List[dict], List[dict]]:
    """Load a behavior concept, returning (train_items, test_items), disjoint,
    shuffled with a fixed seed (mirrors the paper's separate generation/test
    splits in App. C.2.1 / Table 6).
    """
    if name in LOCAL_CONCEPT_FILES:
        items = _load_jsonl(os.path.join(os.path.dirname(__file__), "data", LOCAL_CONCEPT_FILES[name]))
    elif name in CONCEPT_FILES:
        items = []
        for rel_path in CONCEPT_FILES[name]:
            local_path = _ensure_downloaded(rel_path)
            items.extend(_load_jsonl(local_path))
    else:
        raise ValueError(f"Unknown concept '{name}'. Options: {list(CONCEPT_FILES) + list(LOCAL_CONCEPT_FILES)}")

    rng = random.Random(seed)
    rng.shuffle(items)

    total_needed = n_train + n_test
    if len(items) < total_needed:
        raise ValueError(f"Concept '{name}' only has {len(items)} items, need {total_needed}")

    train_items = items[:n_train]
    test_items = items[n_train:total_needed]
    return train_items, test_items
