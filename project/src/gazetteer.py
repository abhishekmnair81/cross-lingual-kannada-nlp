"""Dravidian Hybrid Gazetteer & Morphological Entity Engine.

Combines verified training-set entity distributions (142k+ tokens, 206k+ phrases)
with Dravidian agglutinative suffix stripping and contextual semantic cues to accurately
identify Person (PER), Location (LOC), and Organization (ORG) in Kannada and Telugu.
"""

from __future__ import annotations

import gzip
import json
import logging
import pickle
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .utils import find_project_root

logger = logging.getLogger("GazetteerEngine")

# Common Kannada & Telugu agglutinative grammatical case suffixes
KANNADA_SUFFIXES = [
    "ನಲ್ಲಿರುವ",
    "ದಲ್ಲಿರುವ",
    "ನಲ್ಲಿ",
    "ದಲ್ಲಿ",
    "ನಿಂದ",
    "ದಿಂದ",
    "ಯಲ್ಲಿ",
    "ಯಿಂದ",
    "ರಿಗೆ",
    "ಯಿಗೆ",
    "ಗಾಗಿ",
    "ಕ್ಕೆ",
    "ಗೆ",
    "ದ",
    "ಯ",
    "ರನ್ನು",
    "ವನ್ನು",
    "ಅನ್ನು",
    "ಅವರು",
    "ಅವರ",
    "ಲ್ಲಿ",
]

TELUGU_SUFFIXES = [
    "లోగల",
    "లోని",
    "లో",
    "నుండి",
    "నుంచి",
    "తో",
    "కి",
    "కు",
    "ని",
    "ను",
    "గారికి",
    "గారు",
    "గా",
]

# Contextual Semantic Indicators
PERSON_CUES_PRE = {
    "ಶ್ರೀ", "ಶ್ರೀಮತಿ", "ಡಾ.", "ಪ್ರೊ.", "ಸಚಿವ", "ಅಧ್ಯಕ್ಷ", "ನಾಯಕ", "ಮುಖ್ಯಮಂತ್ರಿ",
    "ಪ್ರಧಾನಿ", "ನಿರ್ದೇಶಕರುಃ", "ಶಾಸಕ", "ಸಂಸದ", "ನಟ", "ನಟಿ",
    "శ్రీ", "శ్రీమతి", "డాక్టర్", "మంత్రి", "ముఖ్యమంత్రి", "నేత", "నాయకుడు", "నటుడు"
}
PERSON_CUES_POST = {
    "ಅವರು", "ಅವರನ್ನ", "ರಾವ್", "ಕುಮಾರ್", "ಸಿಂಗ್", "ಗೌಡ", "ಶೆಟ್ಟಿ", "ಭಟ್", "ಹೆಗಡೆ",
    "ರೆಡ್ಡಿ", "నాయుడు", "గారు", "రావు", "రెడ్డి", "కుమార్", "సింగ్"
}

ORG_CUES = {
    "ಸಂಸ್ಥೆ", "ಸಂಸ್ಥೆಯಲ್ಲಿ", "ಕಂಪನಿ", "ಕಚೇರಿ", "ಬ್ಯಾಂಕ್", "ವಿಶ್ವವಿದ್ಯಾಲಯ", "ಪಕ್ಷ",
    "මණ್ಡಳಿ", "ಇಲಾಖೆ", "ಸಮಿತಿ", "ಟ್ರಸ್ಟ್", "ಶಾಲೆ", "ಕಾಲೇಜು",
    "కార్యాలయం", "సంస్థ", "బ్యాంకు", "పార్టీ", "సమితి", "కంపెనీ", "లిమిటెడ్", "ట్రస్ట్", "కళాశాల"
}

LOC_CUES = {
    "ನಗರ", "ಜಿಲ್ಲೆ", "ರಾಜ್ಯ", "ದೇಶ", "ತಾಲೂಕು", "ಗ್ರಾಮ", "ಕೇಂದ್ರ", "ಬಡಾವಣೆ", "ರಸ್ತೆ",
    "నగరం", "జిల్లా", "రాష్ట్రం", "ప్రాంతం", "గ్రామం", "దేశం", "రోడ్డు"
}

# Singleton memory cache
_GAZETTEER_CACHE: Optional[Dict[str, Any]] = None


def normalize(text: str) -> str:
    """Normalize unicode and strip punctuation for dictionary matching."""
    norm = unicodedata.normalize("NFC", str(text).strip())
    # Strip common punctuation attached to words
    norm = norm.rstrip(".,;:!?\"'()[]{}«»")
    norm = norm.lstrip(".,;:!?\"'()[]{}«»")
    return norm


def get_gazetteer() -> Dict[str, Any]:
    """Load gazetteer into memory."""
    global _GAZETTEER_CACHE
    if _GAZETTEER_CACHE is not None:
        return _GAZETTEER_CACHE

    proj_root = find_project_root()
    outputs_dir = proj_root / "outputs"
    pkl_file = outputs_dir / "gazetteer.pkl.gz"
    json_file = outputs_dir / "gazetteer.json"

    if pkl_file.exists():
        try:
            with gzip.open(pkl_file, "rb") as f:
                _GAZETTEER_CACHE = pickle.load(f)
                return _GAZETTEER_CACHE
        except Exception as e:
            logger.warning(f"Failed to load pkl.gz gazetteer: {e}")

    if json_file.exists():
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                _GAZETTEER_CACHE = json.load(f)
                return _GAZETTEER_CACHE
        except Exception as e:
            logger.warning(f"Failed to load json gazetteer: {e}")

    _GAZETTEER_CACHE = {"tokens": {}, "phrases": {}}
    return _GAZETTEER_CACHE


def strip_agglutinative_suffix(word: str) -> List[str]:
    """Generate candidate stems by peeling common Dravidian suffixes."""
    candidates = [word]
    all_suffixes = sorted(KANNADA_SUFFIXES + TELUGU_SUFFIXES, key=len, reverse=True)

    for suffix in all_suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            stem = word[:-len(suffix)]
            candidates.append(stem)
            # Kannada sandhi vowel restore (e.g. ಬೆಂಗಳೂರಿನ -> ಬೆಂಗಳೂರು)
            if stem.endswith("ಿನ"):
                candidates.append(stem[:-2] + "ು")
            if stem.endswith("ನ"):
                candidates.append(stem[:-1] + "ು")
            if stem.endswith("ಿ"):
                candidates.append(stem[:-1] + "ು")

    return candidates


def lookup_token(token: str, gazetteer: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    """Lookup token or its morphological base in gazetteer.
    
    Returns:
        (entity_type, confidence) or None.
    """
    clean_tok = normalize(token)
    if not clean_tok or len(clean_tok) < 2:
        return None

    tokens_db = gazetteer.get("tokens", {})

    # 1. Direct match
    if clean_tok in tokens_db:
        entry = tokens_db[clean_tok]
        if entry["count"] >= 1 and entry["ratio"] >= 0.5:
            conf = min(0.98, 0.85 + (entry["count"] / 50.0) * 0.1)
            return entry["type"], conf

    # 2. Agglutinative stem match
    stems = strip_agglutinative_suffix(clean_tok)
    for stem in stems[1:]:
        if stem in tokens_db:
            entry = tokens_db[stem]
            if entry["count"] >= 1 and entry["ratio"] >= 0.5:
                conf = min(0.95, 0.82 + (entry["count"] / 50.0) * 0.1)
                return entry["type"], conf

    return None


def extract_entities_with_gazetteer(
    tokens: List[str],
    neural_tags: List[str],
    neural_confs: List[float],
) -> Tuple[List[str], List[float]]:
    """Combine neural predictions with high-coverage gazetteer and linguistics."""
    gazetteer = get_gazetteer()
    phrases_db = gazetteer.get("phrases", {})
    num_tokens = len(tokens)

    refined_tags = list(neural_tags)
    refined_confs = list(neural_confs)

    # 1. Multi-word phrase matching (sliding window: len 4, 3, 2)
    matched_indices: Set[int] = set()

    for window_size in range(4, 1, -1):
        for i in range(num_tokens - window_size + 1):
            if any(idx in matched_indices for idx in range(i, i + window_size)):
                continue

            sub_tokens = [normalize(tokens[j]) for j in range(i, i + window_size)]
            phrase_str = " ".join(sub_tokens)

            if phrase_str in phrases_db:
                ent_type = phrases_db[phrase_str]["type"]
                refined_tags[i] = f"B-{ent_type}"
                refined_confs[i] = 0.96
                matched_indices.add(i)

                for k in range(1, window_size):
                    refined_tags[i + k] = f"I-{ent_type}"
                    refined_confs[i + k] = 0.95
                    matched_indices.add(i + k)

    # 2. Single token gazetteer & morphological lookup
    for i in range(num_tokens):
        if i in matched_indices:
            continue

        raw_tok = tokens[i]
        match = lookup_token(raw_tok, gazetteer)

        if match:
            ent_type, conf = match
            # Check if this continues previous entity of same type
            if i > 0 and refined_tags[i - 1].endswith(f"-{ent_type}"):
                refined_tags[i] = f"I-{ent_type}"
            else:
                refined_tags[i] = f"B-{ent_type}"
            refined_confs[i] = conf
            matched_indices.add(i)

    # 3. Contextual linguistic cues
    for i in range(num_tokens):
        if i in matched_indices:
            continue

        raw_tok = tokens[i]
        clean_tok = normalize(raw_tok)

        # Preceding person honorific / title (e.g. "ನಿರ್ದೇಶಕರುಃ ಶಿವಗಣೇಶ್", "ಶ್ರೀ ರಾಹುಲ್")
        if i > 0 and normalize(tokens[i - 1]) in PERSON_CUES_PRE and len(clean_tok) > 2:
            refined_tags[i] = "B-PER"
            refined_confs[i] = 0.92
            matched_indices.add(i)
            continue

        # Following person suffix cue (e.g. "ರಾಹುಲ್ ಅವರು")
        if i < num_tokens - 1 and normalize(tokens[i + 1]) in PERSON_CUES_POST and len(clean_tok) > 2:
            refined_tags[i] = "B-PER"
            refined_confs[i] = 0.90
            matched_indices.add(i)
            continue

        # Following organization cue (e.g. "ಇನ್ಫೋಸಿಸ್ ಸಂಸ್ಥೆಯಲ್ಲಿ", "ಗೂಗಲ್ ಕಚೇರಿ")
        if i < num_tokens - 1 and normalize(tokens[i + 1]) in ORG_CUES and len(clean_tok) > 2:
            refined_tags[i] = "B-ORG"
            refined_confs[i] = 0.91
            matched_indices.add(i)
            continue

        # Following location cue (e.g. "ಬೆಂಗಳೂರು ನಗರ")
        if i < num_tokens - 1 and normalize(tokens[i + 1]) in LOC_CUES and len(clean_tok) > 2:
            refined_tags[i] = "B-LOC"
            refined_confs[i] = 0.91
            matched_indices.add(i)
            continue

    # Default unassigned tokens to 'O' with high confidence
    for i in range(num_tokens):
        if refined_tags[i] == "O":
            refined_confs[i] = max(0.92, refined_confs[i])

    return refined_tags, refined_confs
