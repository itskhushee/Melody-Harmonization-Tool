"""End-to-end harmonization pipeline."""

from __future__ import annotations
import pickle
from harmonizer.hmm import HMM
from harmonizer.chord_vocab import CHORD_VOCAB
from harmonizer.midi_io import load_melody


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
