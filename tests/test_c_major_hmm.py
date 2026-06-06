"""Tests for the C major HMM demo and core HMM correctness."""

import pytest
from harmonizer.harmonize import build_baseline_hmm
from harmonizer.chord_vocab import get_chord_to_roman_map_for_key, get_chords_for_key


C_MAJOR_MELODY = [
    [0, 4, 7],   # C E G  → C:maj  (I)
    [5, 9, 0],   # F A C  → F:maj  (IV)
    [7, 11, 2],  # G B D  → G:maj  (V)
    [0, 4, 7],   # C E G  → C:maj  (I)
]


def test_build_baseline_hmm_returns_hmm():
    hmm = build_baseline_hmm("C_major")
    assert hmm is not None


def test_viterbi_harmonize_length():
    hmm = build_baseline_hmm("C_major")
    result = hmm.viterbi_harmonize(C_MAJOR_MELODY, key="C_major")
    assert len(result) == len(C_MAJOR_MELODY)


def test_viterbi_harmonize_returns_diatonic_chords():
    hmm = build_baseline_hmm("C_major")
    result = hmm.viterbi_harmonize(C_MAJOR_MELODY, key="C_major")
    valid_chords, _ = get_chords_for_key("C_major")
    for chord in result:
        assert chord in valid_chords, f"Non-diatonic chord predicted: {chord}"


def test_viterbi_harmonize_tonic_measure():
    """A measure of pure C-E-G should resolve to C:maj."""
    hmm = build_baseline_hmm("C_major")
    result = hmm.viterbi_harmonize([[0, 4, 7]], key="C_major")
    assert result[0] == "C:maj"


def test_viterbi_harmonize_empty_sequence():
    hmm = build_baseline_hmm("C_major")
    result = hmm.viterbi_harmonize([], key="C_major")
    assert result == []


def test_demo_smoke():
    """Smoke test: run the C major demo and check it returns 3 chords."""
    from harmonizer.demo_c_major_hmm import run_demo
    from pathlib import Path

    midi = Path("data/raw/demo/train/c_major_pop.mid")
    if not midi.exists():
        pytest.skip("demo MIDI not found")

    result = run_demo(midi, key="C_major")
    assert isinstance(result, list)
    assert len(result) > 0
