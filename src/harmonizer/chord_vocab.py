"""Chord vocabulary — defines the hidden states of the HMM.

Each chord is represented as a root pitch class (0–11) plus a quality.
The full vocabulary covers all 12 roots × 3 triad qualities
(major, minor, diminished) plus a no-chord token "N".

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

QUALITIES = ["maj", "min", "dim"]

# Full chord vocabulary: 12 roots × 3 qualities + no-chord token = 37 states
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

# ── Key / mode data ───────────────────────────────────────────────────────────

MAJOR_SCALE_INTERVALS: list[int] = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE_INTERVALS: list[int] = [0, 2, 3, 5, 7, 8, 10]

# Diatonic triad quality per scale degree (I–VII)
MAJOR_DIATONIC_QUALITIES: list[str] = ["maj", "min", "min", "maj", "maj", "min", "dim"]
MINOR_DIATONIC_QUALITIES: list[str] = ["min", "dim", "maj", "min", "maj", "maj", "maj"]

MAJOR_ROMAN_NUMERALS: list[str] = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
MINOR_ROMAN_NUMERALS: list[str] = ["i", "ii°", "III", "iv", "V", "VI", "VII"]

# Common functional chord progressions expressed as Roman numerals
MAJOR_ROMAN_TRANSITIONS: dict[str, list[str]] = {
    "I":    ["IV", "V", "vi"],
    "ii":   ["V"],
    "iii":  ["vi", "IV"],
    "IV":   ["V", "I"],
    "V":    ["I", "vi"],
    "vi":   ["IV", "ii", "V", "I"],
    "vii°": ["I"],
}

MINOR_ROMAN_TRANSITIONS: dict[str, list[str]] = {
    "i":   ["iv", "V", "VI"],
    "ii°": ["V"],
    "III": ["VI", "iv"],
    "iv":  ["V", "i"],
    "V":   ["i", "VI"],
    "VI":  ["iv", "ii°", "V", "i"],
    "VII": ["III"],
}


# ── Key helpers ───────────────────────────────────────────────────────────────

def _parse_key(key: str) -> tuple[int, str]:
    """Parse 'G_major' → (7, 'major') or 'F#_minor' → (6, 'minor')."""
    root_str, mode = key.rsplit("_", 1)
    if root_str not in ROOTS:
        raise ValueError(f"Unknown root: {root_str!r}")
    if mode not in ("major", "minor"):
        raise ValueError(f"Unknown mode: {mode!r} — expected 'major' or 'minor'")
    return ROOTS.index(root_str), mode


def get_chords_for_key(key: str) -> tuple[list[str], set[int]]:
    """Return diatonic chord labels and scale pitch classes for any key.

    Args:
        key: e.g. "C_major", "G_major", "F#_minor"

    Returns:
        (chord_labels, scale_pcs) — 7 diatonic chords + set of 7 pitch classes.
    """
    root_pc, mode = _parse_key(key)
    intervals = MAJOR_SCALE_INTERVALS if mode == "major" else MINOR_SCALE_INTERVALS
    qualities = MAJOR_DIATONIC_QUALITIES if mode == "major" else MINOR_DIATONIC_QUALITIES

    scale_pcs = {(root_pc + i) % 12 for i in intervals}
    chord_labels = [
        f"{ROOTS[(root_pc + i) % 12]}:{q}"
        for i, q in zip(intervals, qualities)
    ]
    return chord_labels, scale_pcs


def get_roman_map_for_key(key: str) -> dict[str, str]:
    """Return Roman numeral → chord label mapping for any key.

    Args:
        key: e.g. "C_major", "D_minor"

    Returns:
        e.g. {"I": "C:maj", "ii": "D:min", ...} for C_major
    """
    root_pc, mode = _parse_key(key)
    intervals = MAJOR_SCALE_INTERVALS if mode == "major" else MINOR_SCALE_INTERVALS
    qualities = MAJOR_DIATONIC_QUALITIES if mode == "major" else MINOR_DIATONIC_QUALITIES
    romans = MAJOR_ROMAN_NUMERALS if mode == "major" else MINOR_ROMAN_NUMERALS

    return {
        roman: f"{ROOTS[(root_pc + i) % 12]}:{q}"
        for roman, i, q in zip(romans, intervals, qualities)
    }


def get_chord_to_roman_map_for_key(key: str) -> dict[str, str]:
    """Return chord label → Roman numeral mapping for any key."""
    return {chord: roman for roman, chord in get_roman_map_for_key(key).items()}


def get_roman_transitions_for_key(key: str) -> dict[str, list[str]]:
    """Return the functional transition table for a key's mode."""
    _, mode = _parse_key(key)
    return MAJOR_ROMAN_TRANSITIONS if mode == "major" else MINOR_ROMAN_TRANSITIONS


# ── Chord utilities ───────────────────────────────────────────────────────────

def pop909_key_to_internal(key_line: str) -> str:
    """Convert a POP909 key_audio.txt line to internal key format.

    Args:
        key_line: e.g. "2.67  191.98  Gb:maj"

    Returns:
        Internal key string e.g. "F#_major"
    """
    raw = key_line.strip().split()[-1]  # last token e.g. "Gb:maj"
    root_str, mode_str = raw.split(":")
    root = POP909_ROOT_MAP.get(root_str, root_str)
    mode = "major" if mode_str == "maj" else "minor"
    return f"{root}_{mode}"


def normalize_pop909_label(raw: str) -> str:
    """Convert a POP909 chord label to internal format.

    Args:
        raw: e.g. "Bb:min7", "C#:7", "N", "G:maj/3"  (slash chords stripped)

    Returns:
        Internal label e.g. "A#:min", "C#:maj", "B:dim", or "N"
    """
    if raw in ("N", "X"):
        return "N"
    raw = raw.split("/")[0]
    root_str, quality_str = raw.split(":")
    root = POP909_ROOT_MAP.get(root_str, root_str)
    quality = POP909_QUALITY_MAP.get(quality_str, "maj")
    return f"{root}:{quality}"


def chord_pitch_classes(chord_label: str) -> set[int]:
    """Return the set of pitch classes (mod 12) for a chord label.

    Args:
        chord_label: e.g. "C:maj", "G:min", or "N"

    Returns:
        Set of pitch class integers (empty set for "N").
    """
    if chord_label == "N":
        return set()
    root_str, quality = chord_label.split(":")
    root_pc = ROOTS.index(root_str)
    return {(root_pc + interval) % 12 for interval in QUALITY_INTERVALS[quality]}


def chord_index(chord_label: str, key: str | None = None) -> int:
    """Return index of a chord in the full vocabulary or a key-specific vocabulary.

    Args:
        chord_label: Internal chord label, such as "C:maj" or "B:dim".
        key: If provided, index into that key's 7-chord list, e.g. "G_major".

    Returns:
        Integer index of the chord.
    """
    if key is None:
        return CHORD_VOCAB.index(chord_label)
    chords, _ = get_chords_for_key(key)
    return chords.index(chord_label)
