"""Learn emission probabilities from free-midi-chords MIDI files.

Parses individual chord voicing MIDI files (triads + 7ths/9ths) and builds
a pitch-class histogram per chord label — P(pitch_class | chord).
"""

from __future__ import annotations
import json
from pathlib import Path

from mido import MidiFile

from harmonizer.chord_vocab import CHORD_VOCAB, get_roman_map_for_key

# Maps each key directory name to its (major_key, minor_key) internal format
_KEY_DIR_MAP: dict[str, tuple[str, str]] = {
    "01 - C Major - A minor":   ("C_major",  "A_minor"),
    "02 - Db Major - Bb minor": ("C#_major", "A#_minor"),
    "03 - D Major - B minor":   ("D_major",  "B_minor"),
    "04 - Eb Major - C minor":  ("D#_major", "C_minor"),
    "05 - E Major - C# minor":  ("E_major",  "C#_minor"),
    "06 - F Major - D minor":   ("F_major",  "D_minor"),
    "07 - Gb Major - Eb minor": ("F#_major", "D#_minor"),
    "08 - G Major - E minor":   ("G_major",  "E_minor"),
    "09 - Ab Major - F minor":  ("G#_major", "F_minor"),
    "10 - A Major - F# minor":  ("A_major",  "F#_minor"),
    "11 - Bb Major - G minor":  ("A#_major", "G_minor"),
    "12 - B Major - G# minor":  ("B_major",  "G#_minor"),
}

# Sub-directories to scan (triads + 7ths give richest pitch class data)
_SCAN_DIRS = ("1 Triad", "2 7th and 9th")


def _extract_pitch_classes(midi_path: Path) -> list[int]:
    """Return all unique pitch classes (0–11) present in a MIDI file."""
    mid = MidiFile(midi_path)
    pcs: set[int] = set()
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                pcs.add(msg.note % 12)
    return list(pcs)


def _roman_from_filename(stem: str) -> str:
    """Extract Roman numeral from a filename like 'I - C' or 'vii - Bdim'.

    Returns the Roman numeral with '°' appended for diminished (vii).
    """
    roman = stem.split(" - ")[0].strip()
    # In our vocab the leading minor-mode diminished is 'ii°', major is 'vii°'
    if roman in ("vii", "ii") and stem.lower().endswith("dim"):
        roman = roman + "°"
    return roman


def learn_emissions(
    midi_base_dir: str,
    smoothing: float = 0.5,
) -> dict:
    """Learn P(pitch_class | chord) from free-midi-chords MIDI files.

    Args:
        midi_base_dir: Path to the extracted midi_files/ directory.
        smoothing:     Laplace smoothing added to every pitch-class count.

    Returns:
        Dict {chord_label: {str(pitch_class): probability}}
        for all 36 non-N chords in CHORD_VOCAB.
    """
    base = Path(midi_base_dir)

    # Initialise counts with Laplace smoothing for every chord × pitch class
    counts: dict[str, dict[str, float]] = {
        chord: {str(pc): smoothing for pc in range(12)}
        for chord in CHORD_VOCAB
        if chord != "N"
    }

    for dir_name, (major_key, minor_key) in _KEY_DIR_MAP.items():
        key_dir = base / dir_name
        if not key_dir.exists():
            continue

        for scan_dir in _SCAN_DIRS:
            for scale_folder, key in (("Major", major_key), ("Minor", minor_key)):
                folder = key_dir / scan_dir / scale_folder
                if not folder.exists():
                    continue

                roman_map = get_roman_map_for_key(key)

                for midi_file in folder.glob("*.mid"):
                    roman = _roman_from_filename(midi_file.stem)

                    # Flexible lookup: try exact, then without °
                    chord_label = roman_map.get(roman) or roman_map.get(roman.rstrip("°"))
                    if chord_label is None:
                        continue

                    for pc in _extract_pitch_classes(midi_file):
                        counts[chord_label][str(pc)] += 1

    # Normalise to probabilities
    probs: dict[str, dict[str, float]] = {}
    for chord, pc_counts in counts.items():
        total = sum(pc_counts.values())
        probs[chord] = {pc: v / total for pc, v in pc_counts.items()}

    return probs


def save_emissions(probs: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(probs, f, indent=2)
    print(f"Emission probabilities saved to {path}")


def load_emissions(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
