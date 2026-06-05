"""Simple MIDI parser for converting melody MIDI into HMM observations.

Output format:
    melody_by_measure = [
        [0, 4, 7],      # measure 1 pitch classes
        [5, 9, 0],      # measure 2 pitch classes
        ...
    ]

Pitch classes:
    C=0, C#=1, D=2, ..., B=11
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from mido import MidiFile


def _get_time_signature(mid: MidiFile) -> tuple[int, int]:
    """Return time signature as numerator, denominator. Defaults to 4/4."""
    for track in mid.tracks:
        for msg in track:
            if msg.type == "time_signature":
                return msg.numerator, msg.denominator
    return 4, 4


def _extract_notes_from_track(track, ignore_drum_channel: bool = True) -> list[tuple[int, int]]:
    """Extract note start events from one MIDI track.

    Returns:
        List of (start_tick, midi_note_number).
    """
    absolute_time = 0
    active_notes: dict[int, list[int]] = defaultdict(list)
    note_starts: list[tuple[int, int]] = []

    for msg in track:
        absolute_time += msg.time

        if ignore_drum_channel and hasattr(msg, "channel") and msg.channel == 9:
            continue

        if msg.type == "note_on" and msg.velocity > 0:
            active_notes[msg.note].append(absolute_time)
            note_starts.append((absolute_time, msg.note))

        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active_notes and active_notes[msg.note]:
                active_notes[msg.note].pop()

    return note_starts


def _choose_melody_track(mid: MidiFile) -> int:
    """Pick a likely melody track.

    Simple heuristic:
    - ignore tracks with no notes
    - prefer tracks with higher average pitch
    """
    best_track_index = 0
    best_score = float("-inf")

    for index, track in enumerate(mid.tracks):
        notes = _extract_notes_from_track(track)

        if not notes:
            continue

        pitches = [pitch for _, pitch in notes]
        avg_pitch = sum(pitches) / len(pitches)

        # Higher notes are more likely to be melody.
        # Add a small note-count factor so tiny tracks are less likely.
        score = avg_pitch + min(len(pitches), 100) * 0.01

        if score > best_score:
            best_score = score
            best_track_index = index

    return best_track_index


def midi_to_melody_by_measure(
    midi_path: str | Path,
    track_index: int | None = None,
    beats_per_measure: int | None = None,
) -> list[list[int]]:
    """Convert a MIDI melody track into measure-level pitch-class observations.

    Args:
        midi_path: Path to a .mid file.
        track_index: Track to parse. If None, a likely melody track is selected.
        beats_per_measure: Defaults to the MIDI time signature numerator, usually 4.

    Returns:
        List of measures, where each measure is a list of pitch classes.
    """
    midi_path = Path(midi_path)
    mid = MidiFile(midi_path)

    numerator, denominator = _get_time_signature(mid)
    beats_per_measure = beats_per_measure or numerator

    # This keeps the first version simple: assume quarter-note beat.
    # For standard 4/4 MIDI files, this is what we want.
    ticks_per_measure = mid.ticks_per_beat * beats_per_measure

    if track_index is None:
        track_index = _choose_melody_track(mid)

    notes = _extract_notes_from_track(mid.tracks[track_index])

    measures: dict[int, list[int]] = defaultdict(list)

    for start_tick, midi_note in notes:
        measure_index = start_tick // ticks_per_measure
        pitch_class = midi_note % 12

        # Avoid duplicate pitch classes inside the same measure.
        if pitch_class not in measures[measure_index]:
            measures[measure_index].append(pitch_class)

    if not measures:
        return []

    max_measure = max(measures.keys())

    return [measures[i] for i in range(max_measure + 1)]


def midi_to_melody_by_beat(
    midi_path: str | Path,
    track_index: int | None = None,
) -> list[list[int]]:
    """Convert a MIDI melody track into beat-level pitch-class observations.

    Same as midi_to_melody_by_measure but groups notes by individual beat
    (ticks_per_beat) instead of full measure. This gives finer granularity
    for chord prediction and better alignment with chord annotations.

    Args:
        midi_path:   Path to a .mid file.
        track_index: Track to parse. If None, a likely melody track is selected.

    Returns:
        List of beats, where each beat is a list of pitch classes.
    """
    midi_path = Path(midi_path)
    mid = MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat

    if track_index is None:
        track_index = _choose_melody_track(mid)

    notes = _extract_notes_from_track(mid.tracks[track_index])

    beats: dict[int, list[int]] = defaultdict(list)
    for start_tick, midi_note in notes:
        beat_index = start_tick // ticks_per_beat
        pitch_class = midi_note % 12
        if pitch_class not in beats[beat_index]:
            beats[beat_index].append(pitch_class)

    if not beats:
        return []

    max_beat = max(beats.keys())
    return [beats[i] for i in range(max_beat + 1)]