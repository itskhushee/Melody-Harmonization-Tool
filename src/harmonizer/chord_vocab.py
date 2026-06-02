"""Chord vocabulary — defines the hidden states of the HMM.

Each chord is represented as a root pitch class (0–11) plus a quality.
The vocabulary can be expanded as needed.
"""

from __future__ import annotations

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES = ["maj", "min", "dom7", "maj7", "min7", "dim", "aug"]

# Full chord vocabulary: all root × quality combinations
CHORD_VOCAB: list[str] = [f"{r}:{q}" for r in ROOTS for q in QUALITIES]

# Pitch classes (semitones above root) for each quality
QUALITY_INTERVALS: dict[str, list[int]] = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "dom7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
}


# Baseline key-specific chord sets for the first HMM demo.
# These use the existing "Root:quality" label format from CHORD_VOCAB.
C_MAJOR_BASELINE_CHORDS: list[str] = ["C:maj", "F:maj", "G:maj", "A:min"]
C_MINOR_BASELINE_CHORDS: list[str] = ["C:min", "F:min", "G:maj", "G#:maj", "A#:maj"]

# Scale pitch classes for scoring melody notes against the selected key.
C_MAJOR_SCALE_PCS: set[int] = {0, 2, 4, 5, 7, 9, 11}
C_MINOR_SCALE_PCS: set[int] = {0, 2, 3, 5, 7, 8, 10, 11}


def get_chords_for_key(key: str) -> tuple[list[str], set[int]]:
    """Return baseline chord labels and scale pitch classes for a supported key.

    Args:
        key: Currently "C_major" or "C_minor".

    Returns:
        A tuple of (chord labels, scale pitch classes).
    """
    if key == "C_major":
        return C_MAJOR_BASELINE_CHORDS, C_MAJOR_SCALE_PCS

    if key == "C_minor":
        return C_MINOR_BASELINE_CHORDS, C_MINOR_SCALE_PCS

    raise ValueError(f"Unsupported key: {key}")


def chord_pitch_classes(chord_label: str) -> set[int]:
    """Return the set of pitch classes (mod 12) for a chord label.

    Args:
        chord_label: e.g. "C:maj" or "G:dom7"

    Returns:
        Set of pitch class integers.
    """
    root_str, quality = chord_label.split(":")
    root_pc = ROOTS.index(root_str)
    return {(root_pc + interval) % 12 for interval in QUALITY_INTERVALS[quality]}


def chord_index(chord_label: str) -> int:
    return CHORD_VOCAB.index(chord_label)
