"""End-to-end harmonization pipeline."""

from __future__ import annotations
import pickle
import numpy as np
from harmonizer.hmm import HMM
from harmonizer.chord_vocab import CHORD_VOCAB, get_chords_for_key, chord_pitch_classes
from harmonizer.midi_io import load_melody


def build_baseline_hmm(key: str) -> tuple[HMM, list[str]]:
    """Build a placeholder HMM for a key using music-theory emission probabilities.

    Parameters (A, pi) are uniform placeholders — they will be replaced by
    train.py once the labeled dataset is ready. Only B is meaningful here.

    Args:
        key: "C_major" or "C_minor".

    Returns:
        Tuple of (HMM, chord_labels) where chord_labels[i] is the name of state i.
    """
    chord_labels, _ = get_chords_for_key(key)
    N = len(chord_labels)
    M = 12  # pitch classes 0–11

    pi = np.full(N, 1.0 / N)
    A  = np.full((N, N), 1.0 / N)

    # Chord tones share 0.9 of the probability; non-chord tones share 0.1
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


def harmonize(model: HMM, midi_path: str) -> list[str]:
    """Run the full pipeline: parse MIDI → Viterbi decode → chord labels.

    Args:
        model: Trained HMM.
        midi_path: Path to a monophonic .mid file.

    Returns:
        List of chord label strings (one per melody note).
    """
    pitches = load_melody(midi_path)
    observations = [p % 12 for p in pitches]
    state_indices = model.viterbi(observations)
    return [CHORD_VOCAB[i] for i in state_indices]
