"""Evaluate chord prediction accuracy against POP909 ground truth annotations."""

from __future__ import annotations
from pathlib import Path

from harmonizer.chord_vocab import normalize_pop909_label, pop909_key_to_internal
from harmonizer.hmm import HMM
from harmonizer.midi_parser import midi_to_melody_by_measure, midi_to_melody_by_beat


def _ground_truth_per_measure(beat_file: Path, chord_file: Path) -> list[str]:
    """Return one ground-truth chord label per measure using downbeat alignment."""
    downbeats = []
    for line in beat_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and float(parts[1]) == 1.0:
            downbeats.append(float(parts[0]))

    annotations = _read_annotations(chord_file)
    return [_chord_at_time(t, annotations) for t in downbeats]


def _ground_truth_per_beat(beat_file: Path, chord_file: Path) -> list[str]:
    """Return one ground-truth chord label per beat."""
    beat_times = []
    for line in beat_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if parts:
            beat_times.append(float(parts[0]))

    annotations = _read_annotations(chord_file)
    return [_chord_at_time(t, annotations) for t in beat_times]


def _read_annotations(chord_file: Path) -> list[tuple[float, float, str]]:
    annotations = []
    for line in chord_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 3:
            annotations.append((float(parts[0]), float(parts[1]), parts[2]))
    return annotations


def _chord_at_time(t: float, annotations: list[tuple[float, float, str]]) -> str:
    for start, end, label in annotations:
        if start <= t < end:
            return normalize_pop909_label(label)
    return "N"


def evaluate(
    pop909_dir: str,
    test_ids: list[str],
    learned_probs: dict | None = None,
    learned_emissions: dict | None = None,
    granularity: str = "beat",
    verbose: bool = False,
) -> dict:
    """Evaluate predicted chords against POP909 ground truth.

    Args:
        pop909_dir:        Path to POP909/POP909/ directory.
        test_ids:          Song IDs to evaluate on.
        learned_probs:     Optional learned transition probabilities.
        learned_emissions: Optional learned emission probabilities.
        granularity:       "measure" (default) or "beat" for finer prediction.
        verbose:           Print per-song results.

    Returns:
        Dict with keys: total, correct, accuracy, skipped.
    """
    base = Path(pop909_dir)
    total = correct = skipped = 0

    for song_id in test_ids:
        song_dir = base / song_id
        try:
            key = pop909_key_to_internal((song_dir / "key_audio.txt").read_text())
            midi_path = song_dir / f"{song_id}.mid"

            if granularity == "beat":
                melody = midi_to_melody_by_beat(midi_path)
                ground_truth = _ground_truth_per_beat(
                    song_dir / "beat_midi.txt",
                    song_dir / "chord_midi.txt",
                )
            else:
                melody = midi_to_melody_by_measure(midi_path)
                ground_truth = _ground_truth_per_measure(
                    song_dir / "beat_midi.txt",
                    song_dir / "chord_midi.txt",
                )

            predicted = HMM.viterbi_harmonize(
                melody, key=key,
                learned_probs=learned_probs,
                learned_emissions=learned_emissions,
            )

            n = min(len(predicted), len(ground_truth))
            song_correct = sum(predicted[i] == ground_truth[i] for i in range(n))
            total   += n
            correct += song_correct

            if verbose:
                print(f"  Song {song_id} ({key}): {song_correct}/{n} = {song_correct/n*100:.0f}%")

        except Exception as e:
            if verbose:
                print(f"  Song {song_id}: skipped ({e})")
            skipped += 1
            continue

    accuracy = correct / total if total > 0 else 0.0
    return {"total": total, "correct": correct, "accuracy": accuracy, "skipped": skipped}
