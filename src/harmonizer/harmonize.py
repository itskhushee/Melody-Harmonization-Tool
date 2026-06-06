"""End-to-end harmonization pipeline."""

from __future__ import annotations
from harmonizer.hmm import HMM
from harmonizer.midi_parser import midi_to_melody_by_beat, midi_to_melody_by_measure
from harmonizer.midi_io import save_harmonized


def harmonize_midi(
    input_path: str,
    output_path: str,
    key: str,
    hmm: HMM,
    granularity: str = "measure",
) -> list[str]:
    """Full pipeline: melody MIDI in → predict chords → two-track MIDI out.

    Args:
        input_path:  Path to a monophonic melody .mid file.
        output_path: Where to write the harmonized .mid file.
        key:         Key to harmonize in, e.g. "C_major", "G_minor".
        hmm:         Trained HMM loaded via HMM.load().
        granularity: "beat" (one chord per beat, more responsive) or
                     "measure" (one chord per measure, smoother/simpler).
                     Use "measure" for simple melodies and demos;
                     "beat" for richer harmonization of longer pieces.

    Returns:
        List of predicted chord labels, one per beat or measure.
    """
    if granularity == "beat":
        melody = midi_to_melody_by_beat(input_path)
    else:
        melody = midi_to_melody_by_measure(input_path)
    chord_labels = hmm.viterbi_harmonize(melody, key=key)
    save_harmonized(output_path, input_path, chord_labels, granularity=granularity)
    return chord_labels
