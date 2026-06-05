"""Evaluate chord prediction accuracy against POP909 ground truth annotations."""

from __future__ import annotations
from pathlib import Path

from harmonizer.chord_vocab import normalize_pop909_label, pop909_key_to_internal
from harmonizer.hmm import HMM
from harmonizer.midi_parser import midi_to_melody_by_measure


def _ground_truth_per_measure(beat_file: Path, chord_file: Path) -> list[str]:
    """Return one ground-truth chord label per measure using downbeat alignment."""
    downbeats = []
    for line in beat_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and float(parts[1]) == 1.0:
            downbeats.append(float(parts[0]))

    annotations = []
    for line in chord_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 3:
            annotations.append((float(parts[0]), float(parts[1]), parts[2]))

    measure_chords = []
    for beat_time in downbeats:
        chord = "N"
        for start, end, label in annotations:
            if start <= beat_time < end:
                chord = normalize_pop909_label(label)
                break
        measure_chords.append(chord)
    return measure_chords


def evaluate(
    pop909_dir: str,
    test_ids: list[str],
    learned_probs: dict | None = None,
    learned_emissions: dict | None = None,
    verbose: bool = False,
) -> dict:
    """Evaluate predicted chords against POP909 ground truth.

    Args:
        pop909_dir:    Path to POP909/POP909/ directory.
        test_ids:      Song IDs to evaluate on.
        learned_probs: Optional learned transition probabilities.
        verbose:       Print per-song results.

    Returns:
        Dict with keys: total, correct, accuracy, skipped.
    """
    base = Path(pop909_dir)
    total = correct = skipped = 0

    for song_id in test_ids:
        song_dir = base / song_id
        try:
            key = pop909_key_to_internal((song_dir / "key_audio.txt").read_text())
            melody = midi_to_melody_by_measure(song_dir / f"{song_id}.mid")
            predicted = HMM.viterbi_harmonize(melody, key=key, learned_probs=learned_probs, learned_emissions=learned_emissions)
            ground_truth = _ground_truth_per_measure(
                song_dir / "beat_midi.txt",
                song_dir / "chord_midi.txt",
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
