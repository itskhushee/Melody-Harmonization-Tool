"""Smoke-test demo: harmonize a melody MIDI file using a theory-based HMM.

Usage:
    PYTHONPATH=src python src/harmonizer/demo_c_major_hmm.py <path/to/melody.mid>
    PYTHONPATH=src python src/harmonizer/demo_c_major_hmm.py data/raw/demo/train/c_major_pop.mid
"""

from __future__ import annotations
import sys
from pathlib import Path

from harmonizer.chord_vocab import get_chord_to_roman_map_for_key
from harmonizer.harmonize import build_baseline_hmm
from harmonizer.midi_parser import midi_to_melody_by_measure


def run_demo(midi_path: str | Path, key: str = "C_major") -> list[str]:
    """Harmonize a MIDI file with a theory-based HMM and print results.

    Args:
        midi_path: Path to a melody .mid file.
        key:       Key to harmonize in, e.g. "C_major", "D_minor".

    Returns:
        Predicted chord labels, one per measure.
    """
    melody = midi_to_melody_by_measure(midi_path)
    print(f"Loaded: {midi_path}  ({len(melody)} measures)")
    print("Melody pitch classes per measure:")
    for i, notes in enumerate(melody, 1):
        print(f"  Measure {i}: {notes}")

    hmm            = build_baseline_hmm(key)
    predicted      = hmm.viterbi_harmonize(melody, key=key)
    chord_to_roman = get_chord_to_roman_map_for_key(key)

    print("\nPredicted chords:  ", " -> ".join(predicted))
    print("Roman numerals:    ", " -> ".join(chord_to_roman.get(c, "?") for c in predicted))
    return predicted


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python demo_c_major_hmm.py <melody.mid> [key]")
        print("  key defaults to C_major")
        sys.exit(1)

    midi_path = Path(sys.argv[1])
    key       = sys.argv[2] if len(sys.argv) > 2 else "C_major"

    if not midi_path.exists():
        print(f"Error: file not found — {midi_path}")
        sys.exit(1)

    run_demo(midi_path, key)


if __name__ == "__main__":
    main()
