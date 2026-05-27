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
