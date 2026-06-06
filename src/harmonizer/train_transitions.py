"""Learn transition probabilities from POP909 chord annotations.

Learns Roman-numeral bigram probabilities separately for major and minor modes,
so the learned weights are key-agnostic and apply to all 24 keys.
"""

from __future__ import annotations
import json
import random
import numpy as np
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
    smoothing: float = 0.1,
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


def learn_pi(
    pop909_dir: str,
    train_ids: list[str],
    smoothing: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Learn initial state distribution from the first chord of each song.

    Returns:
        (pi_major, pi_minor) — each shape (7,), normalised to sum to 1.
    """
    import numpy as np
    base = Path(pop909_dir)

    major_counts = {r: smoothing for r in MAJOR_ROMAN_NUMERALS}
    minor_counts = {r: smoothing for r in MINOR_ROMAN_NUMERALS}

    for song_id in train_ids:
        song_dir = base / song_id
        try:
            key = pop909_key_to_internal((song_dir / "key_audio.txt").read_text())
            chord_to_roman = get_chord_to_roman_map_for_key(key)
            chords = _read_chord_sequence(song_dir / "chord_midi.txt")
            _, mode = key.rsplit("_", 1)
            counts = major_counts if mode == "major" else minor_counts

            for chord in chords:
                if chord != "N" and chord in chord_to_roman:
                    roman = chord_to_roman[chord]
                    if roman in counts:
                        counts[roman] += 1
                    break
        except Exception:
            continue

    pi_major = np.array([major_counts[r] for r in MAJOR_ROMAN_NUMERALS])
    pi_minor = np.array([minor_counts[r] for r in MINOR_ROMAN_NUMERALS])
    return pi_major / pi_major.sum(), pi_minor / pi_minor.sum()


def build_A_matrices(probs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Convert learned_probs dict to numpy A matrices.

    Returns:
        (A_major, A_minor) — each shape (7, 7).
    """
    import numpy as np
    A_major = np.array([
        [probs["major"][r1][r2] for r2 in MAJOR_ROMAN_NUMERALS]
        for r1 in MAJOR_ROMAN_NUMERALS
    ])
    A_minor = np.array([
        [probs["minor"][r1][r2] for r2 in MINOR_ROMAN_NUMERALS]
        for r1 in MINOR_ROMAN_NUMERALS
    ])
    return A_major, A_minor


def train_hmm(
    pop909_dir: str,
    train_ids: list[str],
    midi_files_dir: str,
    smoothing: float = 1.0,
) -> "HMM":
    """Train a full HMM from POP909 (pi + A) and free-midi-chords (B).

    Args:
        pop909_dir:     Path to POP909/POP909/ directory.
        train_ids:      Song IDs to train on.
        midi_files_dir: Path to free-midi-chords midi_files/ directory.
        smoothing:      Laplace smoothing for counts.

    Returns:
        Trained HMM instance with pi, A, B properly populated.
    """
    from harmonizer.hmm import HMM
    from harmonizer.train_emissions import learn_emissions

    print("Learning pi and A from POP909...")
    probs              = learn_transitions(pop909_dir, train_ids, smoothing)
    pi_major, pi_minor = learn_pi(pop909_dir, train_ids, smoothing)
    A_major,  A_minor  = build_A_matrices(probs)

    print("Learning B from free-midi-chords...")
    free_midi_emissions = learn_emissions(midi_files_dir)

    print("Learning B from POP909 melody data...")
    from harmonizer.train_emissions import learn_emissions_from_pop909, combine_emissions
    pop909_emissions = learn_emissions_from_pop909(pop909_dir, train_ids)

    print("Combining emissions (50% POP909, 50% free-midi-chords)...")
    emissions = combine_emissions(free_midi_emissions, pop909_emissions, pop909_weight=0.5)

    return HMM(pi_major, pi_minor, A_major, A_minor, emissions)


def save_transitions(probs: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(probs, f, indent=2)
    print(f"Transition probabilities saved to {path}")


def load_transitions(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
