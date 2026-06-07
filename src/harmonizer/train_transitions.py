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


def _chord_at_time(t: float, annotations: list[tuple[float, float, str]]) -> str:
    for start, end, label in annotations:
        if start <= t < end:
            return normalize_pop909_label(label)
    return "N"


def _read_annotations(chord_file: Path) -> list[tuple[float, float, str]]:
    annotations = []
    for line in chord_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 3:
            try:
                annotations.append((float(parts[0]), float(parts[1]), parts[2]))
            except ValueError:
                continue
    return annotations


def learn_transitions(
    pop909_dir: str,
    train_ids: list[str],
    smoothing: float = 0.1,
) -> dict:
    """Learn BEAT-LEVEL Roman numeral transition probabilities from POP909.

    Counts bigrams between consecutive beats (using beat_midi.txt for timing
    and chord_midi.txt for annotations). Because most consecutive beats share
    the same chord, self-transitions are naturally represented — fixing the
    critical bug where the old segment-level approach never counted them,
    causing Viterbi to actively penalise chord persistence.

    Args:
        pop909_dir: Path to the POP909/POP909/ directory.
        train_ids:  Song IDs to learn from.
        smoothing:  Laplace smoothing count added to every bigram.

    Returns:
        Dict with keys "major" and "minor", each a nested dict
        probs[prev_roman][next_roman] = probability.
    """
    base = Path(pop909_dir)

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
            key             = pop909_key_to_internal((song_dir / "key_audio.txt").read_text())
            chord_to_roman  = get_chord_to_roman_map_for_key(key)
            _, mode         = key.rsplit("_", 1)
            counts          = major_counts if mode == "major" else minor_counts
            annotations     = _read_annotations(song_dir / "chord_midi.txt")

            beat_times = [
                float(l.split()[0])
                for l in (song_dir / "beat_midi.txt").read_text().strip().splitlines()
                if l.strip()
            ]

            # Beat-level chord sequence (includes self-transitions)
            beat_chords = [_chord_at_time(bt, annotations) for bt in beat_times]
            romans = [
                chord_to_roman.get(c)
                for c in beat_chords
            ]

            for prev, nxt in zip(romans, romans[1:]):
                if prev and nxt and prev in counts and nxt in counts[prev]:
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
    """Learn initial state distribution from the first in-key chord of each song.

    Uses beat-level timing: scans all beats in order and counts the first
    beat that has a diatonic (in-key) chord. This is robust to songs whose
    chord annotations start slightly after beat 0 (which shows up as 'N').

    Returns:
        (pi_major, pi_minor) — each shape (7,), normalised to sum to 1.
    """
    base = Path(pop909_dir)

    major_counts = {r: smoothing for r in MAJOR_ROMAN_NUMERALS}
    minor_counts = {r: smoothing for r in MINOR_ROMAN_NUMERALS}

    for song_id in train_ids:
        song_dir = base / song_id
        try:
            key            = pop909_key_to_internal((song_dir / "key_audio.txt").read_text())
            chord_to_roman = get_chord_to_roman_map_for_key(key)
            _, mode        = key.rsplit("_", 1)
            counts         = major_counts if mode == "major" else minor_counts
            annotations    = _read_annotations(song_dir / "chord_midi.txt")

            beat_times = [
                float(l.split()[0])
                for l in (song_dir / "beat_midi.txt").read_text().strip().splitlines()
                if l.strip()
            ]

            for bt in beat_times:
                chord = _chord_at_time(bt, annotations)
                roman = chord_to_roman.get(chord)
                if roman and roman in counts:
                    counts[roman] += 1
                    break

        except Exception:
            continue

    pi_major = np.array([major_counts[r] for r in MAJOR_ROMAN_NUMERALS])
    pi_minor = np.array([minor_counts[r] for r in MINOR_ROMAN_NUMERALS])
    return pi_major / pi_major.sum(), pi_minor / pi_minor.sum()


def _infer_key(chord_sequence: list[str]) -> str | None:
    """Infer the most likely key for a chord sequence by diatonic coverage.

    Tries all 24 major/minor keys and returns the one in which the greatest
    fraction of observed chords are diatonic. Returns None if the sequence
    is empty or no key covers at least half the chords.
    """
    from harmonizer.chord_vocab import get_chords_for_key, ROOTS

    if not chord_sequence:
        return None

    best_key: str | None = None
    best_count = 0

    for root in ROOTS:
        for mode in ("major", "minor"):
            key = f"{root}_{mode}"
            diatonic, _ = get_chords_for_key(key)
            count = sum(1 for c in chord_sequence if c in diatonic)
            if count > best_count:
                best_count = count
                best_key = key

    if best_count < len(chord_sequence) * 0.4:
        return None

    return best_key


def learn_transitions_from_nottingham(
    nottingham_midi_dir: str,
    counts: dict[str, dict[str, dict[str, float]]],
) -> None:
    """Add Nottingham beat-level chord bigrams to existing transition counts in-place.

    Args:
        nottingham_midi_dir: Path to the Nottingham MIDI/ directory.
        counts: Mutable dict with keys "major" and "minor", each a nested dict
                counts[mode][prev_roman][next_roman]. Modified in-place.
    """
    from harmonizer.parse_nottingham import load_all_nottingham
    from harmonizer.chord_vocab import get_chord_to_roman_map_for_key

    corpus = load_all_nottingham(nottingham_midi_dir)
    added = 0

    for _, chord_seq in corpus:
        key = _infer_key(chord_seq)
        if key is None:
            continue

        _, mode = key.rsplit("_", 1)
        chord_to_roman = get_chord_to_roman_map_for_key(key)
        mode_counts = counts[mode]

        romans = [chord_to_roman.get(c) for c in chord_seq]

        for prev, nxt in zip(romans, romans[1:]):
            if prev and nxt and prev in mode_counts and nxt in mode_counts[prev]:
                mode_counts[prev][nxt] += 1
                added += 1

    print(f"  Added {added:,} Nottingham bigrams from {len(corpus)} songs")


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
    nottingham_midi_dir: str | None = None,
    smoothing: float = 0.1,
) -> "HMM":
    """Train a full HMM from POP909, free-midi-chords, and optionally Nottingham.

    Args:
        pop909_dir:          Path to POP909/POP909/ directory.
        train_ids:           Song IDs to train on.
        midi_files_dir:      Path to free-midi-chords directory.
        nottingham_midi_dir: Path to Nottingham MIDI/ directory (optional).
        smoothing:           Laplace smoothing for counts.

    Returns:
        Trained HMM instance with pi, A, B properly populated.
    """
    from harmonizer.hmm import HMM
    from harmonizer.train_emissions import (
        learn_emissions,
        learn_emissions_bass_filtered,
        learn_emissions_from_nottingham,
        combine_emissions,
    )

    # ── Transitions ──────────────────────────────────────────────────────────
    print("Learning pi and A from POP909 (beat-level)...")
    probs = learn_transitions(pop909_dir, train_ids, smoothing)
    pi_major, pi_minor = learn_pi(pop909_dir, train_ids, smoothing)

    A_major, A_minor = build_A_matrices(probs)

    # ── Emissions ─────────────────────────────────────────────────────────────
    print("Learning B from free-midi-chords...")
    free_midi_emissions = learn_emissions(midi_files_dir)

    print("Learning B from POP909 (bass-confirmed beats only)...")
    pop909_emissions = learn_emissions_bass_filtered(pop909_dir, train_ids)

    if nottingham_midi_dir:
        print("Learning B from Nottingham melody-over-chord pairs...")
        nott_emissions = learn_emissions_from_nottingham(nottingham_midi_dir)
        # Three-way blend tuned via grid search: 65% POP909, 20% Nottingham, 15% free-midi-chords.
        # Nottingham transitions are intentionally excluded — folk chord grammar
        # is different enough from pop to hurt POP909 test accuracy.
        all_chords = set(pop909_emissions) | set(nott_emissions) | set(free_midi_emissions)
        emissions = {
            chord: {
                pc: 0.65 * pop909_emissions.get(chord, {}).get(pc, 0.0)
                  + 0.20 * nott_emissions.get(chord, {}).get(pc, 0.0)
                  + 0.15 * free_midi_emissions.get(chord, {}).get(pc, 0.0)
                for pc in (
                    set(pop909_emissions.get(chord, {}))
                    | set(nott_emissions.get(chord, {}))
                    | set(free_midi_emissions.get(chord, {}))
                )
            }
            for chord in all_chords
        }
    else:
        print("Combining emissions (70% POP909, 30% free-midi-chords)...")
        emissions = combine_emissions(free_midi_emissions, pop909_emissions, pop909_weight=0.7)

    return HMM(pi_major, pi_minor, A_major, A_minor, emissions)


def save_transitions(probs: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(probs, f, indent=2)
    print(f"Transition probabilities saved to {path}")


def load_transitions(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
