"""End-to-end harmonization pipeline."""

from __future__ import annotations
import numpy as np
from harmonizer.hmm import HMM
from harmonizer.midi_parser import midi_to_melody_by_measure
from harmonizer.midi_io import save_harmonized
from harmonizer.chord_vocab import (
    get_chords_for_key,
    chord_pitch_classes,
    MAJOR_ROMAN_NUMERALS,
    MINOR_ROMAN_NUMERALS,
)


def build_baseline_hmm(key: str) -> HMM:
    """Build a music-theory HMM with no training data required.

    Uses uniform pi and A matrices and theory-based emissions:
    chord tones get 0.9 of the probability mass, non-chord tones share 0.1.
    Useful for demos and as a fallback when no trained model exists.

    Args:
        key: e.g. "C_major", "D_minor"

    Returns:
        An HMM instance ready for viterbi_harmonize().
    """
    _, mode = key.rsplit("_", 1)
    N = 7

    pi_uniform = np.full(N, 1.0 / N)
    A_uniform  = np.full((N, N), 1.0 / N)

    # Build theory-based emissions for every chord in the 37-label full vocab
    from harmonizer.chord_vocab import CHORD_VOCAB
    emissions: dict[str, dict[str, float]] = {}
    for label in CHORD_VOCAB:
        if label == "N":
            continue
        in_chord  = chord_pitch_classes(label)
        out_chord = set(range(12)) - in_chord
        probs: dict[str, float] = {}
        for pc in in_chord:
            probs[str(pc)] = 0.9 / len(in_chord)
        for pc in out_chord:
            probs[str(pc)] = 0.1 / len(out_chord)
        emissions[label] = probs

    return HMM(
        pi_major  = pi_uniform.copy(),
        pi_minor  = pi_uniform.copy(),
        A_major   = A_uniform.copy(),
        A_minor   = A_uniform.copy(),
        emissions = emissions,
    )


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
    melody       = midi_to_melody_by_measure(input_path)
    chord_labels = hmm.viterbi_harmonize(melody, key=key)
    save_harmonized(output_path, input_path, chord_labels, granularity="measure")
    return chord_labels
