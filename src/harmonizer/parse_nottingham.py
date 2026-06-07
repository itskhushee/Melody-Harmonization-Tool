"""Parse Nottingham MIDI files into (melody_by_beat, chord_by_beat) pairs.

Nottingham MIDI files have exactly two tracks:
  Track 0 — monophonic melody (single notes at varying beat offsets)
  Track 1 — chord accompaniment (3 simultaneous notes, always on beat boundaries)

Chords are identified by matching the observed pitch-class set against every
chord in CHORD_VOCAB and picking the one with the highest Jaccard overlap.
"""

from __future__ import annotations
from collections import defaultdict
from pathlib import Path

from mido import MidiFile

from harmonizer.chord_vocab import CHORD_VOCAB, QUALITY_INTERVALS, ROOTS


# Pre-compute pitch-class sets for every chord in the vocabulary (except N)
_CHORD_PCS: dict[str, frozenset[int]] = {}
for _label in CHORD_VOCAB:
    if _label == "N":
        continue
    _root_str, _quality = _label.split(":")
    _root_pc = ROOTS.index(_root_str)
    _chord_pcs = frozenset((_root_pc + i) % 12 for i in QUALITY_INTERVALS[_quality])
    _CHORD_PCS[_label] = _chord_pcs


def _identify_chord(pitch_classes: frozenset[int]) -> str | None:
    """Return the best-matching chord label for a set of pitch classes.

    Uses Jaccard similarity: |intersection| / |union|.
    Returns None if no chord has any overlap with the observed pitch classes.
    """
    if not pitch_classes:
        return None

    best_label: str | None = None
    best_score = -1.0

    for label, chord_pcs in _CHORD_PCS.items():
        intersection = len(pitch_classes & chord_pcs)
        if intersection == 0:
            continue
        union = len(pitch_classes | chord_pcs)
        score = intersection / union
        if score > best_score:
            best_score = score
            best_label = label

    return best_label


def load_nottingham_song(
    midi_path: str | Path,
) -> tuple[list[list[int]], list[str | None]]:
    """Parse one Nottingham MIDI file into per-beat melody and chord sequences.

    Args:
        midi_path: Path to a .mid file from the Nottingham dataset.

    Returns:
        (melody_by_beat, chord_by_beat) where:
          melody_by_beat[t] = list of pitch classes (0–11) for melody notes
                              that start on beat t.
          chord_by_beat[t]  = chord label string (or None if no chord on that beat).
        Both lists have the same length (max_beat + 1).
    """
    mid = MidiFile(midi_path)
    tpb = mid.ticks_per_beat

    if len(mid.tracks) < 2:
        return [], []

    melody_track = mid.tracks[0]
    chord_track  = mid.tracks[1]

    # ── Extract melody: one note at a time (monophonic) ─────────────────────
    melody_by_beat: dict[int, list[int]] = defaultdict(list)
    abs_tick = 0
    for msg in melody_track:
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            beat = abs_tick // tpb
            pc   = msg.note % 12
            if pc not in melody_by_beat[beat]:
                melody_by_beat[beat].append(pc)

    # ── Extract chords: group simultaneous notes by exact tick ───────────────
    # Notes within the same tick are part of the same chord voicing.
    chord_notes_by_tick: dict[int, set[int]] = defaultdict(set)
    abs_tick = 0
    for msg in chord_track:
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            chord_notes_by_tick[abs_tick].add(msg.note % 12)

    # Map each chord tick to a beat and identify the chord
    chord_by_beat: dict[int, str | None] = {}
    for tick, pcs in chord_notes_by_tick.items():
        if len(pcs) < 2:   # skip single notes (not a chord)
            continue
        beat  = tick // tpb
        label = _identify_chord(frozenset(pcs))
        if label:
            chord_by_beat[beat] = label

    # ── Align into parallel lists ────────────────────────────────────────────
    max_beat = max(
        (max(melody_by_beat.keys(), default=-1)),
        (max(chord_by_beat.keys(),  default=-1)),
    )
    if max_beat < 0:
        return [], []

    melody_seq = [melody_by_beat.get(t, [])   for t in range(max_beat + 1)]
    chord_seq  = [chord_by_beat.get(t)         for t in range(max_beat + 1)]

    return melody_seq, chord_seq


def load_all_nottingham(
    midi_dir: str | Path,
) -> list[tuple[list[list[int]], list[str]]]:
    """Load all Nottingham MIDI files and return paired (melody, chord) sequences.

    Beats with no chord annotation are forward-filled from the previous beat
    so every melody beat has an associated chord label.

    Args:
        midi_dir: Path to the Nottingham MIDI/ directory.

    Returns:
        List of (melody_by_beat, chord_by_beat) pairs. Only songs where at
        least 4 beats have both melody notes and a chord label are included.
    """
    corpus: list[tuple[list[list[int]], list[str]]] = []
    midi_dir = Path(midi_dir)

    for midi_path in sorted(midi_dir.glob("*.mid")):
        try:
            melody_seq, chord_seq = load_nottingham_song(midi_path)
        except Exception:
            continue

        if not chord_seq:
            continue

        # Forward-fill chord annotations
        filled_chords: list[str] = []
        last_chord = "C:maj"   # sensible default if file starts with no chord
        for chord in chord_seq:
            if chord is not None:
                last_chord = chord
            filled_chords.append(last_chord)

        # Keep only songs with at least 4 beats of paired data
        paired = sum(
            1 for m, c in zip(melody_seq, filled_chords)
            if m and c
        )
        if paired >= 4:
            corpus.append((melody_seq, filled_chords))

    return corpus
