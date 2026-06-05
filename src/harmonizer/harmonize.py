"""End-to-end harmonization pipeline."""

from __future__ import annotations
from harmonizer.hmm import HMM
from harmonizer.midi_parser import midi_to_melody_by_measure
from harmonizer.midi_io import save_harmonized


def harmonize_midi(
    input_path: str,
    output_path: str,
    key: str,
    hmm: HMM,
) -> list[str]:
    """Full pipeline: melody MIDI in → predict chords → two-track MIDI out.

    Args:
        input_path:  Path to a monophonic melody .mid file.
        output_path: Where to write the harmonized .mid file.
        key:         Key to harmonize in, e.g. "C_major", "G_minor".
        hmm:         Trained HMM loaded via HMM.load().

    Returns:
        List of predicted chord labels, one per measure.
    """
    melody      = midi_to_melody_by_measure(input_path)
    chord_labels = hmm.viterbi_harmonize(melody, key=key)
    save_harmonized(output_path, input_path, chord_labels, granularity="measure")
    return chord_labels
