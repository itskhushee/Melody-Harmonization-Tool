"""MIDI I/O — load a melody and write a harmonized two-track MIDI."""

from __future__ import annotations
from mido import MidiFile, MidiTrack, Message
from harmonizer.chord_vocab import chord_pitch_classes, ROOTS
from harmonizer.midi_parser import _choose_melody_track, _get_time_signature


def load_melody(path: str) -> list[int]:
    """Parse a MIDI file and return melody pitches in onset order.

    Args:
        path: Path to the .mid file.

    Returns:
        List of integer MIDI pitches (0–127).
    """
    import pretty_midi
    midi = pretty_midi.PrettyMIDI(path)
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        notes.extend(instrument.notes)
    notes.sort(key=lambda n: n.start)
    return [n.pitch for n in notes]


def _voice_chord(root_pc: int, pcs: set[int], base: int = 48) -> list[int]:
    """Return MIDI note numbers for a chord, voiced above the root in closed position.

    Args:
        root_pc: Root pitch class (0–11).
        pcs:     All pitch classes in the chord.
        base:    MIDI note for octave floor (default 48 = C3).

    Returns:
        Sorted list of MIDI note numbers.
    """
    notes = []
    for pc in sorted(pcs):
        note = base + pc
        if pc < root_pc:
            note += 12
        notes.append(note)
    return sorted(notes)


def _compress_chords(
    labels: list[str],
    ticks_per_slot: int,
    min_beats: int = 2,
    max_beats: int = 8,
) -> list[tuple[str, int, int]]:
    """Merge consecutive identical labels and enforce min/max chord durations.

    Returns a list of (label, start_tick, end_tick) segments.

    - Consecutive identical chords are merged into one long segment.
    - Segments shorter than min_beats × ticks_per_slot are absorbed into the
      preceding chord (prevents single-beat flicker).
    - Segments longer than max_beats × ticks_per_slot are split into equal
      halves (prevents the model staying on the tonic for many measures).
    """
    from itertools import groupby

    # Step 1: merge consecutive identical labels into runs
    runs: list[tuple[str, int]] = [
        (label, sum(1 for _ in group))
        for label, group in groupby(labels)
    ]

    # Step 2: absorb short runs into the preceding chord
    min_ticks = min_beats * ticks_per_slot
    merged: list[tuple[str, int]] = []
    for label, count in runs:
        dur = count * ticks_per_slot
        if merged and dur < min_ticks and label != "N":
            prev_label, prev_dur = merged[-1]
            merged[-1] = (prev_label, prev_dur + dur)
        else:
            merged.append((label, dur))

    # Step 3: split runs that exceed max_beats into equal halves
    max_ticks = max_beats * ticks_per_slot
    split: list[tuple[str, int]] = []
    for label, dur in merged:
        while dur > max_ticks and label != "N":
            half = (dur // 2 // ticks_per_slot) * ticks_per_slot
            if half < min_ticks:
                break
            split.append((label, half))
            dur -= half
        split.append((label, dur))

    # Step 4: convert to (label, start_tick, end_tick)
    segments: list[tuple[str, int, int]] = []
    tick = 0
    for label, dur in split:
        segments.append((label, tick, tick + dur))
        tick += dur
    return segments


def save_harmonized(
    output_path: str,
    input_midi_path: str,
    chord_labels: list[str],
    granularity: str = "beat",
) -> None:
    """Write a two-track MIDI: original melody + chord accompaniment.

    Track 0 — original melody (copied from input).
    Track 1 — predicted chords, with consecutive identical chords merged into
               single held events and very short flicker-changes absorbed into
               the surrounding chord. This makes the output sound like a real
               accompaniment rather than repeating 1-beat stabs.

    Args:
        output_path:     Where to write the .mid file.
        input_midi_path: Original melody MIDI (used for melody track + timing).
        chord_labels:    Chord labels from viterbi_harmonize().
        granularity:     "beat" (one slot per beat) or "measure" (one per measure).
    """
    original = MidiFile(input_midi_path)
    out = MidiFile(ticks_per_beat=original.ticks_per_beat, type=1)

    # ── Track 0: copy original melody ────────────────────────────────────────
    melody_idx = _choose_melody_track(original)
    out.tracks.append(original.tracks[melody_idx])

    # ── Track 1: chord accompaniment ─────────────────────────────────────────
    numerator, _ = _get_time_signature(original)
    ticks_per_beat  = original.ticks_per_beat
    ticks_per_slot  = ticks_per_beat if granularity == "beat" else ticks_per_beat * numerator

    segments = _compress_chords(chord_labels, ticks_per_slot, min_beats=2)

    chord_track = MidiTrack()
    out.tracks.append(chord_track)
    chord_track.append(Message("program_change", program=0, channel=1, time=0))

    prev_tick = 0
    for label, start_tick, end_tick in segments:
        if label == "N":
            prev_tick = end_tick
            continue

        root_str = label.split(":")[0]
        root_pc  = ROOTS.index(root_str)
        pcs      = chord_pitch_classes(label)
        notes    = _voice_chord(root_pc, pcs)

        for j, note in enumerate(notes):
            delta = (start_tick - prev_tick) if j == 0 else 0
            chord_track.append(Message("note_on", note=note, velocity=60, channel=1, time=delta))
            prev_tick = start_tick

        for j, note in enumerate(notes):
            delta = (end_tick - prev_tick) if j == 0 else 0
            chord_track.append(Message("note_off", note=note, velocity=0, channel=1, time=delta))
            prev_tick = end_tick

    out.save(output_path)
