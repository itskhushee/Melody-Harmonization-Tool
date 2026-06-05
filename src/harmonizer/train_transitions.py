"""Learn transition probabilities from POP909 chord annotations.

Learns Roman-numeral bigram probabilities separately for major and minor modes,
so the learned weights are key-agnostic and apply to all 24 keys.
"""

from __future__ import annotations
import json
import random
from pathlib import Path

from harmonizer.chord_vocab import (
    normalize_pop909_label,
    pop909_key_to_internal,
    get_chord_to_roman_map_for_key,
    MAJOR_ROMAN_NUMERALS,
    MINOR_ROMAN_NUMERALS,
)


def get_train_test_split(
    pop909_dir: str,
    train_ratio: float = 0.88,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Return (train_ids, test_ids) shuffled with a fixed seed.

    Args:
        pop909_dir: Path to the POP909/POP909/ directory.
        train_ratio: Fraction of songs to use for training.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_ids, test_ids).
    """
    base = Path(pop909_dir)
    all_ids = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name.isdigit())
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    split = int(len(all_ids) * train_ratio)
    return all_ids[:split], all_ids[split:]


def _read_chord_sequence(chord_file: Path) -> list[str]:
    """Read normalized chord labels from a chord_midi.txt file (order only, no timing)."""
    chords = []
    for line in chord_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 3:
            chords.append(normalize_pop909_label(parts[2]))
    return chords


def learn_transitions(
    pop909_dir: str,
    train_ids: list[str],
    smoothing: float = 1.0,
) -> dict:
    """Learn Roman numeral transition probabilities from POP909 chord sequences.

    Args:
        pop909_dir: Path to the POP909/POP909/ directory.
        train_ids:  Song IDs to learn from.
        smoothing:  Laplace smoothing count added to every bigram.

    Returns:
        Dict with keys "major" and "minor", each a nested dict
        probs[prev_roman][next_roman] = probability.
    """
    base = Path(pop909_dir)

    # Initialize counts with Laplace smoothing
    major_counts: dict[str, dict[str, float]] = {
        r1: {r2: smoothing for r2 in MAJOR_ROMAN_NUMERALS}
        for r1 in MAJOR_ROMAN_NUMERALS
    }
    minor_counts: dict[str, dict[str, float]] = {
        r1: {r2: smoothing for r2 in MINOR_ROMAN_NUMERALS}
        for r1 in MINOR_ROMAN_NUMERALS
    }

    skipped = 0
    for song_id in train_ids:
        song_dir = base / song_id
        try:
            key = pop909_key_to_internal((song_dir / "key_audio.txt").read_text())
            chord_to_roman = get_chord_to_roman_map_for_key(key)
            chords = _read_chord_sequence(song_dir / "chord_midi.txt")

            # Map to Roman numerals, skip N and out-of-key chords
            romans = [
                chord_to_roman[c]
                for c in chords
                if c != "N" and c in chord_to_roman
            ]

            _, mode = key.rsplit("_", 1)
            counts = major_counts if mode == "major" else minor_counts

            for prev, nxt in zip(romans, romans[1:]):
                if prev in counts and nxt in counts[prev]:
                    counts[prev][nxt] += 1

        except Exception:
            skipped += 1
            continue

    if skipped:
        print(f"  Skipped {skipped} songs (missing files or parse errors)")

    def _normalize(counts: dict) -> dict:
        return {
            r1: {r2: v / sum(row.values()) for r2, v in row.items()}
            for r1, row in counts.items()
        }

    return {"major": _normalize(major_counts), "minor": _normalize(minor_counts)}


def save_transitions(probs: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(probs, f, indent=2)
    print(f"Transition probabilities saved to {path}")


def load_transitions(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
