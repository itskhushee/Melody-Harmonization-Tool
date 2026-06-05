"""End-to-end harmonization pipeline."""

from __future__ import annotations
import pickle
import numpy as np
from harmonizer.hmm import HMM
from harmonizer.chord_vocab import get_chords_for_key, chord_pitch_classes
from harmonizer.midi_parser import midi_to_melody_by_measure, midi_to_melody_by_beat
from harmonizer.midi_io import save_harmonized


def harmonize_midi(
    input_path: str,
    output_path: str,
    key: str = "C_major",
    learned_probs: dict | None = None,
    learned_emissions: dict | None = None,
) -> list[str]:
    """Full pipeline: melody MIDI in → predict chords → two-track MIDI out.

    Predicts one chord per measure and writes block chords to the output MIDI.
    For accuracy evaluation at finer granularity use evaluate.py directly.

    Args:
        input_path:        Path to a monophonic melody .mid file.
        output_path:       Where to write the harmonized .mid file.
        key:               Key to harmonize in, e.g. "C_major", "G_minor".
        learned_probs:     Optional data-driven transition probabilities.
        learned_emissions: Optional data-driven emission probabilities.

    Returns:
        List of predicted chord labels, one per measure.
    """
    melody = midi_to_melody_by_measure(input_path)
    chord_labels = HMM.viterbi_harmonize(
        melody, key=key,
        learned_probs=learned_probs,
        learned_emissions=learned_emissions,
    )
    save_harmonized(output_path, input_path, chord_labels, granularity="measure")
    return chord_labels


# ── Utilities kept for future trained-model support ───────────────────────────

def build_baseline_hmm(key: str) -> tuple[HMM, list[str]]:
    """Build a placeholder HMM using music-theory emission probabilities.

    A and pi are uniform placeholders; only B is meaningful here.
    Replace with estimate_parameters() once a labeled corpus is ready.
    """
    chord_labels, _ = get_chords_for_key(key)
    N = len(chord_labels)
    M = 12

    pi = np.full(N, 1.0 / N)
    A  = np.full((N, N), 1.0 / N)

    B = np.zeros((N, M))
    for i, label in enumerate(chord_labels):
        in_chord  = chord_pitch_classes(label)
        out_chord = set(range(M)) - in_chord
        for pc in in_chord:
            B[i, pc] = 0.9 / len(in_chord)
        for pc in out_chord:
            B[i, pc] = 0.1 / len(out_chord)

    return HMM(pi=pi, A=A, B=B), chord_labels


def load_model(path: str) -> HMM:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model: HMM, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)
