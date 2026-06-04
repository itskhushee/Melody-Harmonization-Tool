from harmonizer.chord_vocab import get_chord_to_roman_map_for_key, COMMON_ROMAN_TRANSITIONS

"""Chord vocabulary — defines the hidden states of the HMM.

Each chord is represented as a root pitch class (0–11) plus a quality.
The full vocabulary covers all 12 roots × 3 triad qualities
(major, minor, diminished) plus a no-chord token "N". For the first HMM
demo, the model should use the key-specific C major chord set rather than
the full vocabulary.

When parsing POP909 annotations, extended qualities are folded into their
closest supported triad quality.

POP909 label → internal label mapping
--------------------------------------
  maj    → maj        major triad
  min    → min        minor triad
  7      → maj        dominant seventh → major
  maj7   → maj        major seventh   → major
  min7   → min        minor seventh   → minor
  sus2   → maj        suspended second → major
  sus4   → maj        suspended fourth → major
  dim    → dim        diminished       → diminished
  hdim7  → dim        half-diminished  → diminished
  dim7   → dim        fully-diminished → diminished
  aug    → maj        augmented        → major
  N      → N          no chord / silence
"""

from __future__ import annotations

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Now supporting major, minor, and diminished triads
QUALITIES = ["maj", "min", "dim"]

# Full chord vocabulary: all root × quality combinations + no-chord token
# 12 roots × 3 qualities = 36 chords + N = 37 states
CHORD_VOCAB: list[str] = [f"{r}:{q}" for r in ROOTS for q in QUALITIES] + ["N"]

# Pitch classes (semitones above root) for each quality
QUALITY_INTERVALS: dict[str, list[int]] = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
}

# POP909 annotation label → internal quality (maj, min, or dim only)
POP909_QUALITY_MAP: dict[str, str] = {
    "maj":   "maj",
    "min":   "min",
    "7":     "maj",
    "maj7":  "maj",
    "min7":  "min",
    "sus2":  "maj",
    "sus4":  "maj",
    "dim":   "dim",
    "hdim7": "dim",
    "dim7":  "dim",
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
        Internal label e.g. "A#:min", "C#:maj", "B:dim", or "N"
    """
    if raw in ("N", "X"):
        return "N"
    # Strip slash (bass note) if present
    raw = raw.split("/")[0]
    root_str, quality_str = raw.split(":")
    root = POP909_ROOT_MAP.get(root_str, root_str)
    quality = POP909_QUALITY_MAP.get(quality_str, "maj")
    return f"{root}:{quality}"


# C major diatonic triads: I, ii, iii, IV, V, vi, vii°
C_MAJOR_BASELINE_CHORDS: list[str] = ["C:maj", "D:min", "E:min", "F:maj", "G:maj", "A:min", "B:dim"]
C_MINOR_BASELINE_CHORDS: list[str] = ["C:min", "D:min", "D#:maj", "F:min", "G:min", "G#:maj", "A#:maj"]

# Scale pitch classes for scoring melody notes against the selected key.
C_MAJOR_SCALE_PCS: set[int] = {0, 2, 4, 5, 7, 9, 11}
C_MINOR_SCALE_PCS: set[int] = {0, 2, 3, 5, 7, 8, 10, 11}

# Roman numeral mappings for the first C major HMM demo.
# These let transition probabilities be defined musically, e.g. I -> IV -> V -> I.
C_MAJOR_ROMAN_TO_CHORD: dict[str, str] = {
    "I": "C:maj",
    "ii": "D:min",
    "iii": "E:min",
    "IV": "F:maj",
    "V": "G:maj",
    "vi": "A:min",
    "vii°": "B:dim",
}

C_MAJOR_CHORD_TO_ROMAN: dict[str, str] = {
    chord: roman for roman, chord in C_MAJOR_ROMAN_TO_CHORD.items()
}


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


# Roman numeral mapping helpers
def get_roman_map_for_key(key: str) -> dict[str, str]:
    """Return Roman numeral to chord-label mapping for a supported key.

    Args:
        key: Currently "C_major".

    Returns:
        A dictionary mapping Roman numerals to internal chord labels.
    """
    if key == "C_major":
        return C_MAJOR_ROMAN_TO_CHORD

    raise ValueError(f"Roman numeral mapping is not implemented for key: {key}")


def get_chord_to_roman_map_for_key(key: str) -> dict[str, str]:
    """Return chord-label to Roman numeral mapping for a supported key.

    Args:
        key: Currently "C_major".

    Returns:
        A dictionary mapping internal chord labels to Roman numerals.
    """
    if key == "C_major":
        return C_MAJOR_CHORD_TO_ROMAN

    raise ValueError(f"Roman numeral mapping is not implemented for key: {key}")


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


def chord_index(chord_label: str, key: str | None = None) -> int:
    """Return index of a chord in the full vocabulary or selected key vocabulary.

    Args:
        chord_label: Internal chord label, such as "C:maj" or "B:dim".
        key: If provided, use the key-specific chord list, e.g. "C_major".

    Returns:
        Integer index of the chord.
    """
    if key is None:
        return CHORD_VOCAB.index(chord_label)

    chords, _ = get_chords_for_key(key)
    return chords.index(chord_label)


COMMON_ROMAN_TRANSITIONS: dict[str, list[str]] = {
    "I": ["IV", "V", "vi"],
    "ii": ["V"],
    "iii": ["vi", "IV"],
    "IV": ["V", "I"],
    "V": ["I", "vi"],
    "vi": ["IV", "ii", "V", "I"],
    "vii°": ["I"],
}


def transition_score(prev_chord: str, next_chord: str, key: str = "C_major") -> float:
    """Score how naturally one chord moves to another chord."""
    chord_to_roman = get_chord_to_roman_map_for_key(key)

    prev_roman = chord_to_roman[prev_chord]
    next_roman = chord_to_roman[next_chord]

    if next_roman in COMMON_ROMAN_TRANSITIONS.get(prev_roman, []):
        return 0.70

    if prev_chord == next_chord:
        return 0.20

    return 0.05
