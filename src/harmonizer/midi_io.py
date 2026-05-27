"""MIDI parsing — converts a MIDI file into a sequence of note observations."""

from __future__ import annotations
import pretty_midi


def load_melody(path: str) -> list[int]:
    """Parse a monophonic MIDI file and return a list of MIDI pitch values.

    Args:
        path: Path to the .mid file.

    Returns:
        List of integer MIDI pitches (0-127) in order of onset time.
    """
    midi = pretty_midi.PrettyMIDI(path)
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        notes.extend(instrument.notes)
    notes.sort(key=lambda n: n.start)
    return [n.pitch for n in notes]


def save_harmonized(path: str, melody_pitches: list[int], chord_labels: list[str]) -> None:
    """Write a two-track MIDI: melody on track 0, chord roots on track 1.

    Args:
        path: Output .mid file path.
        melody_pitches: Original melody MIDI pitches.
        chord_labels: Chord label per melody note (same length as melody_pitches).
    """
    raise NotImplementedError
