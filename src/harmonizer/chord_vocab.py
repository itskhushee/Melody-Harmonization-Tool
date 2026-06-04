"""Chord vocabulary — defines the hidden states of the HMM.

Each chord is represented as a root pitch class (0–11) plus a quality.
The vocabulary covers all 12 keys × 2 qualities (24 chords) plus a
no-chord token "N", for a total of 25 states.

Only major and minor triads are used. When parsing POP909 annotations,
all other qualities are folded into their closest major/minor equivalent.

POP909 label → internal label mapping
--------------------------------------
  maj    → maj        major triad
  min    → min        minor triad
  7      → maj        dominant seventh → major
  maj7   → maj        major seventh   → major
  min7   → min        minor seventh   → minor
  sus2   → maj        suspended second → major
  sus4   → maj        suspended fourth → major
  dim    → min        diminished       → minor
  hdim7  → min        half-diminished  → minor
  dim7   → min        fully-diminished → minor
  aug    → maj        augmented        → major
  N      → N          no chord / silence
"""

from __future__ import annotations

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

QUALITIES = ["maj", "min"]

# Full chord vocabulary: all root × quality combinations + no-chord token
# 12 roots × 2 qualities = 24 chords + N = 25 states
CHORD_VOCAB: list[str] = [f"{r}:{q}" for r in ROOTS for q in QUALITIES] + ["N"]

# Pitch classes (semitones above root) for each quality
QUALITY_INTERVALS: dict[str, list[int]] = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
}

# POP909 annotation label → internal quality (maj or min only)
POP909_QUALITY_MAP: dict[str, str] = {
    "maj":   "maj",
    "min":   "min",
    "7":     "maj",
    "maj7":  "maj",
    "min7":  "min",
    "sus2":  "maj",
    "sus4":  "maj",
    "dim":   "min",
    "hdim7": "min",
    "dim7":  "min",
    "aug":   "maj",
}

# Enharmonic root spellings used in POP909 → canonical root
POP909_ROOT_MAP: dict[str, str] = {
    "Cb": "B",  "Db": "C#", "Eb": "D#",
    "Fb": "E",  "Gb": "F#", "Ab": "G#",
    "Bb": "A#", "B#": "C",  "E#": "F",
}


def normalize_pop909_label(raw: str) -> str:
    """Convert a POP909 chord label to internal format.

    Args:
        raw: e.g. "Bb:min7", "C#:7", "N", "G:maj/3"  (slash chords stripped)

    Returns:
        Internal label e.g. "A#:min7", "C#:dom7", "N"
    """
    if raw in ("N", "X"):
        return "N"
    # Strip slash (bass note) if present
    raw = raw.split("/")[0]
    root_str, quality_str = raw.split(":")
    root = POP909_ROOT_MAP.get(root_str, root_str)
    quality = POP909_QUALITY_MAP.get(quality_str, "maj")
    return f"{root}:{quality}"


# Baseline key-specific chord sets for the first HMM demo.
# These use the existing "Root:quality" label format from CHORD_VOCAB.
C_MAJOR_BASELINE_CHORDS: list[str] = ["C:maj", "D:min", "E:min", "F:maj", "G:maj", "A:min", "B:min"]
C_MINOR_BASELINE_CHORDS: list[str] = ["C:min", "D:min", "D#:maj", "F:min", "G:min", "G#:maj", "A#:maj"]

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
        chord_label: e.g. "C:maj", "G:dom7", or "N"

    Returns:
        Set of pitch class integers (empty set for "N").
    """
    if chord_label == "N":
        return set()
    root_str, quality = chord_label.split(":")
    root_pc = ROOTS.index(root_str)
    return {(root_pc + interval) % 12 for interval in QUALITY_INTERVALS[quality]}


def chord_index(chord_label: str) -> int:
    return CHORD_VOCAB.index(chord_label)
