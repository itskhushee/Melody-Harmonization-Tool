"""Extended unit tests for core harmonizer components.

Covers:
  - HMM._smooth_predictions  (the blip-removal post-processor)
  - HMM._fill_empty_beats    (sparse-melody filler)
  - HMM.save / HMM.load      (round-trip persistence)
  - chord_vocab helpers       (chord_pitch_classes, get_chords_for_key,
                               normalize_pop909_label, pop909_key_to_internal)
  - viterbi_harmonize         (emission-driven output, key invariants)
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from harmonizer.hmm import HMM
from harmonizer.harmonize import build_baseline_hmm
from harmonizer.chord_vocab import (
    CHORD_VOCAB,
    chord_pitch_classes,
    get_chords_for_key,
    get_roman_map_for_key,
    normalize_pop909_label,
    pop909_key_to_internal,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_hmm() -> HMM:
    """Baseline (uniform) HMM — no training data needed."""
    return build_baseline_hmm("C_major")


# ═══════════════════════════════════════════════════════════════════════════════
# HMM._smooth_predictions
# ═══════════════════════════════════════════════════════════════════════════════

class TestSmoothPredictions:

    def test_collapses_single_blip(self):
        """A→B→A should collapse to A→A→A."""
        result = HMM._smooth_predictions(["C:maj", "D:min", "C:maj"])
        assert result == ["C:maj", "C:maj", "C:maj"]

    def test_preserves_real_change(self):
        """A→B→C is a genuine chord change — must not be modified."""
        seq = ["C:maj", "F:maj", "G:maj"]
        assert HMM._smooth_predictions(seq) == seq

    def test_preserves_longer_run(self):
        """Three consecutive different chords should be left alone."""
        seq = ["C:maj", "C:maj", "G:maj", "G:maj"]
        assert HMM._smooth_predictions(seq) == seq

    def test_collapses_multiple_blips(self):
        """Two independent blips with different sandwiching chords both collapse."""
        # ["C:maj", "D:min", "C:maj", "G:maj", "F:maj", "G:maj"]
        # pos 1: D:min sandwiched by C:maj → C:maj
        # pos 4: F:maj sandwiched by G:maj → G:maj
        seq = ["C:maj", "D:min", "C:maj", "G:maj", "F:maj", "G:maj"]
        result = HMM._smooth_predictions(seq)
        assert result[1] == "C:maj"
        assert result[4] == "G:maj"

    def test_cascade_collapse_documented_behavior(self):
        """The while-changed loop causes cascading collapse of alternating patterns.

        This is the known '_smooth_predictions' behavior that causes the
        all-tonic output bug documented in docs/progress_log.md (2026-06-09).

        A→B→A→B→A runs of alternating chords collapse to all-A, because each
        pass through the loop makes previously independent sandwiches eligible.
        """
        # After pass 1: G:maj at positions 1 and 3 are each sandwiched by
        # C:maj → both collapse.  The sequence degenerates to all-C:maj.
        seq = ["C:maj", "G:maj", "C:maj", "G:maj", "C:maj"]
        result = HMM._smooth_predictions(seq)
        assert all(c == "C:maj" for c in result)

    def test_empty_list(self):
        assert HMM._smooth_predictions([]) == []

    def test_single_element(self):
        assert HMM._smooth_predictions(["C:maj"]) == ["C:maj"]



# ═══════════════════════════════════════════════════════════════════════════════
# HMM._fill_empty_beats
# ═══════════════════════════════════════════════════════════════════════════════

class TestFillEmptyBeats:

    def test_no_empty_beats_unchanged(self):
        melody = [[0, 4, 7], [5, 9], [7, 11, 2]]
        assert HMM._fill_empty_beats(melody) == melody

    def test_borrows_from_left(self):
        """Empty beat after a non-empty beat borrows from the left."""
        melody = [[0, 4, 7], [], [7, 11, 2]]
        filled = HMM._fill_empty_beats(melody)
        assert filled[1] == [0, 4, 7]

    def test_borrows_from_right_when_no_left(self):
        """Empty beat at position 0 borrows from the first non-empty beat to the right."""
        melody = [[], [5, 9, 0], [7, 11, 2]]
        filled = HMM._fill_empty_beats(melody)
        assert filled[0] == [5, 9, 0]


# ═══════════════════════════════════════════════════════════════════════════════
# chord_pitch_classes
# ═══════════════════════════════════════════════════════════════════════════════

class TestChordPitchClasses:

    def test_c_major(self):
        assert chord_pitch_classes("C:maj") == {0, 4, 7}

    def test_g_major(self):
        assert chord_pitch_classes("G:maj") == {7, 11, 2}

    def test_d_minor(self):
        assert chord_pitch_classes("D:min") == {2, 5, 9}

    def test_b_diminished(self):
        assert chord_pitch_classes("B:dim") == {11, 2, 5}


# ═══════════════════════════════════════════════════════════════════════════════
# get_chords_for_key
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetChordsForKey:

    def test_c_major_chords(self):
        chords, _ = get_chords_for_key("C_major")
        assert chords == ["C:maj", "D:min", "E:min", "F:maj", "G:maj", "A:min", "B:dim"]

    def test_always_seven_chords(self):
        for root in ["C", "D", "E", "F", "G", "A", "B"]:
            for mode in ["major", "minor"]:
                key = f"{root}_{mode}"
                chords, _ = get_chords_for_key(key)
                assert len(chords) == 7, f"{key} returned {len(chords)} chords"

    def test_no_duplicate_chords_in_key(self):
        chords, _ = get_chords_for_key("G_major")
        assert len(chords) == len(set(chords)), "Duplicate chords in G_major"

    def test_all_chords_in_vocab(self):
        """Every chord returned by get_chords_for_key must be in CHORD_VOCAB."""
        for root in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]:
            for mode in ["major", "minor"]:
                chords, _ = get_chords_for_key(f"{root}_{mode}")
                for chord in chords:
                    assert chord in CHORD_VOCAB, f"{chord} not in CHORD_VOCAB"


# ═══════════════════════════════════════════════════════════════════════════════
# normalize_pop909_label
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizePop909Label:

    def test_n_label(self):
        assert normalize_pop909_label("N") == "N"

    def test_x_label(self):
        assert normalize_pop909_label("X") == "N"

    def test_major(self):
        assert normalize_pop909_label("C:maj") == "C:maj"

    def test_minor(self):
        assert normalize_pop909_label("A:min") == "A:min"

    def test_dominant_seventh_folds_to_major(self):
        assert normalize_pop909_label("C#:7") == "C#:maj"

    def test_minor_seventh_folds_to_minor(self):
        assert normalize_pop909_label("Bb:min7") == "A#:min"


# ═══════════════════════════════════════════════════════════════════════════════
# pop909_key_to_internal
# ═══════════════════════════════════════════════════════════════════════════════

class TestPop909KeyToInternal:

    def test_c_major(self):
        text = "0.0 30.0 C:maj\n"
        assert pop909_key_to_internal(text) == "C_major"

    def test_g_major(self):
        text = "0.0 60.0 G:maj\n"
        assert pop909_key_to_internal(text) == "G_major"

    def test_enharmonic_flat(self):
        """Bb should be mapped to A#_major."""
        text = "0.0 45.0 Bb:maj\n"
        assert pop909_key_to_internal(text) == "A#_major"


# ═══════════════════════════════════════════════════════════════════════════════
# HMM save / load round-trip
# ═══════════════════════════════════════════════════════════════════════════════

class TestHMMSaveLoad:

    def test_round_trip_preserves_pi(self):
        hmm = _make_hmm()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        hmm.save(path)
        loaded = HMM.load(path)
        np.testing.assert_allclose(loaded.pi_major, hmm.pi_major)
        np.testing.assert_allclose(loaded.pi_minor, hmm.pi_minor)

    def test_round_trip_preserves_A(self):
        hmm = _make_hmm()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        hmm.save(path)
        loaded = HMM.load(path)
        np.testing.assert_allclose(loaded.A_major, hmm.A_major)
        np.testing.assert_allclose(loaded.A_minor, hmm.A_minor)



# ═══════════════════════════════════════════════════════════════════════════════
# Emission probability sanity checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmissionProbabilities:

    def test_baseline_emissions_sum_to_one(self):
        """Each chord's emission distribution must sum to ~1.0."""
        hmm = _make_hmm()
        for chord, probs in hmm.emissions.items():
            if chord == "N":
                continue
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-9, (
                f"Emissions for {chord} sum to {total:.6f}, expected 1.0"
            )

    def test_chord_tones_have_higher_emission(self):
        """In baseline HMM, chord tones must have higher probability than non-chord tones."""
        hmm = _make_hmm()
        pcs = chord_pitch_classes("C:maj")   # {0, 4, 7}
        non_pcs = set(range(12)) - pcs
        for ct in pcs:
            for nct in non_pcs:
                assert hmm.emissions["C:maj"][str(ct)] > hmm.emissions["C:maj"][str(nct)]


# ═══════════════════════════════════════════════════════════════════════════════
# viterbi_harmonize — additional invariants
# ═══════════════════════════════════════════════════════════════════════════════

class TestViterbiHarmonize:

    def test_g_major_returns_g_major_diatonic_only(self):
        """All predicted chords must be diatonic to the requested key."""
        hmm = _make_hmm()
        melody = [[7, 11, 2], [5, 9], [2, 5, 9], [7, 11, 2]]
        result = hmm.viterbi_harmonize(melody, key="G_major")
        valid, _ = get_chords_for_key("G_major")
        for chord in result:
            assert chord in valid, f"Non-diatonic chord for G_major: {chord}"

    def test_output_length_matches_input(self):
        hmm = _make_hmm()
        for length in [1, 3, 7, 16]:
            melody = [[0, 4, 7]] * length
            result = hmm.viterbi_harmonize(melody, key="C_major")
            assert len(result) == length

    def test_single_measure_tonic_notes(self):
        hmm = _make_hmm()
        result = hmm.viterbi_harmonize([[0, 4, 7]], key="C_major")
        assert result == ["C:maj"]

    def test_d_minor_key_output_diatonic(self):
        """Spot-check a minor key — all output chords must be diatonic to D_minor."""
        hmm = _make_hmm()
        melody = [[2, 5, 9], [0, 4], [9, 0, 4], [2, 5, 9]]
        result = hmm.viterbi_harmonize(melody, key="D_minor")
        valid, _ = get_chords_for_key("D_minor")
        for chord in result:
            assert chord in valid, f"Non-diatonic chord for D_minor: {chord}"


# ═══════════════════════════════════════════════════════════════════════════════
# get_roman_map_for_key
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetRomanMapForKey:

    def test_c_major_roman_map(self):
        mapping = get_roman_map_for_key("C_major")
        assert mapping["I"] == "C:maj"
        assert mapping["IV"] == "F:maj"
        assert mapping["V"] == "G:maj"
        assert mapping["vi"] == "A:min"
