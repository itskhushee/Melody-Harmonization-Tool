"""Learn emission probabilities from free-midi-chords MIDI files.

Parses individual chord voicing MIDI files (triads + 7ths/9ths) and builds
a pitch-class histogram per chord label — P(pitch_class | chord).
"""

from __future__ import annotations
import json
from pathlib import Path

from mido import MidiFile

from harmonizer.chord_vocab import CHORD_VOCAB, get_roman_map_for_key

# Maps each key directory name to its (major_key, minor_key) internal format
_KEY_DIR_MAP: dict[str, tuple[str, str]] = {
    "01 - C Major - A minor":   ("C_major",  "A_minor"),
    "02 - Db Major - Bb minor": ("C#_major", "A#_minor"),
    "03 - D Major - B minor":   ("D_major",  "B_minor"),
    "04 - Eb Major - C minor":  ("D#_major", "C_minor"),
    "05 - E Major - C# minor":  ("E_major",  "C#_minor"),
    "06 - F Major - D minor":   ("F_major",  "D_minor"),
    "07 - Gb Major - Eb minor": ("F#_major", "D#_minor"),
    "08 - G Major - E minor":   ("G_major",  "E_minor"),
    "09 - Ab Major - F minor":  ("G#_major", "F_minor"),
    "10 - A Major - F# minor":  ("A_major",  "F#_minor"),
    "11 - Bb Major - G minor":  ("A#_major", "G_minor"),
    "12 - B Major - G# minor":  ("B_major",  "G#_minor"),
}

# Sub-directories to scan (triads + 7ths give richest pitch class data)
_SCAN_DIRS = ("1 Triad", "2 7th and 9th")


def _extract_pitch_classes(midi_path: Path) -> list[int]:
    """Return all unique pitch classes (0–11) present in a MIDI file."""
    mid = MidiFile(midi_path)
    pcs: set[int] = set()
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                pcs.add(msg.note % 12)
    return list(pcs)


def _roman_from_filename(stem: str) -> str:
    """Extract Roman numeral from a filename like 'I - C' or 'vii - Bdim'.

    Returns the Roman numeral with '°' appended for diminished (vii).
    """
    roman = stem.split(" - ")[0].strip()
    # In our vocab the leading minor-mode diminished is 'ii°', major is 'vii°'
    if roman in ("vii", "ii") and stem.lower().endswith("dim"):
        roman = roman + "°"
    return roman


def learn_emissions(
    midi_base_dir: str,
    smoothing: float = 0.5,
) -> dict:
    """Learn P(pitch_class | chord) from free-midi-chords MIDI files.

    Args:
        midi_base_dir: Path to the extracted midi_files/ directory.
        smoothing:     Laplace smoothing added to every pitch-class count.

    Returns:
        Dict {chord_label: {str(pitch_class): probability}}
        for all 36 non-N chords in CHORD_VOCAB.
    """
    base = Path(midi_base_dir)

    # Initialise counts with Laplace smoothing for every chord × pitch class
    counts: dict[str, dict[str, float]] = {
        chord: {str(pc): smoothing for pc in range(12)}
        for chord in CHORD_VOCAB
        if chord != "N"
    }

    for dir_name, (major_key, minor_key) in _KEY_DIR_MAP.items():
        key_dir = base / dir_name
        if not key_dir.exists():
            continue

        for scan_dir in _SCAN_DIRS:
            for scale_folder, key in (("Major", major_key), ("Minor", minor_key)):
                folder = key_dir / scan_dir / scale_folder
                if not folder.exists():
                    continue

                roman_map = get_roman_map_for_key(key)

                for midi_file in folder.glob("*.mid"):
                    roman = _roman_from_filename(midi_file.stem)

                    # Flexible lookup: try exact, then without °
                    chord_label = roman_map.get(roman) or roman_map.get(roman.rstrip("°"))
                    if chord_label is None:
                        continue

                    for pc in _extract_pitch_classes(midi_file):
                        counts[chord_label][str(pc)] += 1

    # Normalise to probabilities
    probs: dict[str, dict[str, float]] = {}
    for chord, pc_counts in counts.items():
        total = sum(pc_counts.values())
        probs[chord] = {pc: v / total for pc, v in pc_counts.items()}

    return probs


def learn_emissions_from_pop909(
    pop909_dir: str,
    train_ids: list[str],
    smoothing: float = 0.5,
) -> dict:
    """Learn P(pitch_class | chord) from POP909 melody notes over chord annotations.

    For each beat in each training song, aligns the melody pitch classes with
    the annotated chord label and counts co-occurrences.

    Args:
        pop909_dir: Path to POP909/POP909/ directory.
        train_ids:  Song IDs to learn from.
        smoothing:  Laplace smoothing added to every pitch-class count.

    Returns:
        Dict {chord_label: {str(pitch_class): probability}}
    """
    from harmonizer.midi_parser import midi_to_melody_by_beat
    from harmonizer.chord_vocab import normalize_pop909_label, pop909_key_to_internal

    base = Path(pop909_dir)

    # Initialise counts with smoothing
    counts: dict[str, dict[str, float]] = {
        chord: {str(pc): smoothing for pc in range(12)}
        for chord in CHORD_VOCAB
        if chord != "N"
    }

    skipped = 0
    for song_id in train_ids:
        song_dir = base / song_id
        try:
            # Melody pitch classes per beat
            melody_by_beat = midi_to_melody_by_beat(song_dir / f"{song_id}.mid")

            # Chord label per beat from annotations
            beat_times = []
            for line in (song_dir / "beat_midi.txt").read_text().strip().splitlines():
                parts = line.strip().split()
                if parts:
                    beat_times.append(float(parts[0]))

            annotations = []
            for line in (song_dir / "chord_midi.txt").read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 3:
                    annotations.append((float(parts[0]), float(parts[1]), parts[2]))

            n = min(len(melody_by_beat), len(beat_times))
            for t in range(n):
                # Find chord at this beat time
                beat_time = beat_times[t]
                chord = "N"
                for start, end, label in annotations:
                    if start <= beat_time < end:
                        chord = normalize_pop909_label(label)
                        break

                if chord == "N" or chord not in counts:
                    continue

                for pc in melody_by_beat[t]:
                    counts[chord][str(pc % 12)] += 1

        except Exception:
            skipped += 1
            continue

    if skipped:
        print(f"  Skipped {skipped} songs in POP909 emission learning")

    # Normalise
    return {
        chord: {pc: v / sum(row.values()) for pc, v in row.items()}
        for chord, row in counts.items()
    }


def learn_emissions_bass_filtered(
    pop909_dir: str,
    train_ids: list[str],
    smoothing: float = 0.1,
) -> dict:
    """Learn P(pitch_class | chord) using only bass-confirmed beats.

    For each beat, the lowest note in the PIANO track (track 3) is the bass.
    We only count melody notes when the bass pitch class matches the annotated
    chord root — these are "solid" beats where the chord is unambiguous,
    giving a cleaner emission signal than using all beats indiscriminately.

    Args:
        pop909_dir: Path to POP909/POP909/ directory.
        train_ids:  Song IDs to learn from.
        smoothing:  Laplace smoothing (lower than default since data is cleaner).

    Returns:
        Dict {chord_label: {str(pitch_class): probability}}
    """
    from collections import defaultdict
    from harmonizer.midi_parser import midi_to_melody_by_beat
    from harmonizer.chord_vocab import normalize_pop909_label, pop909_key_to_internal, ROOTS

    base = Path(pop909_dir)

    counts: dict[str, dict[str, float]] = {
        chord: {str(pc): smoothing for pc in range(12)}
        for chord in CHORD_VOCAB
        if chord != "N"
    }

    skipped = 0
    for song_id in train_ids:
        song_dir = base / song_id
        try:
            mid = MidiFile(song_dir / f"{song_id}.mid")
            tpb = mid.ticks_per_beat

            # Melody pitch classes per beat (track 1)
            melody_by_beat = midi_to_melody_by_beat(song_dir / f"{song_id}.mid",
                                                     track_index=1)

            # Bass note per beat: lowest note in PIANO track (track 3)
            abs_tick = 0
            piano_notes_by_beat: dict[int, list[int]] = defaultdict(list)
            for msg in mid.tracks[3]:
                abs_tick += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    piano_notes_by_beat[abs_tick // tpb].append(msg.note)

            beat_times = [
                float(l.split()[0])
                for l in (song_dir / "beat_midi.txt").read_text().strip().splitlines()
                if l.strip()
            ]
            annotations = [
                (float(p[0]), float(p[1]), p[2])
                for l in (song_dir / "chord_midi.txt").read_text().strip().splitlines()
                if (p := l.strip().split()) and len(p) == 3
            ]

            n = min(len(melody_by_beat), len(beat_times))
            for t in range(n):
                bt = beat_times[t]
                chord = "N"
                for start, end, label in annotations:
                    if start <= bt < end:
                        chord = normalize_pop909_label(label)
                        break

                if chord == "N" or chord not in counts:
                    continue

                piano_notes = piano_notes_by_beat.get(t, [])
                if not piano_notes:
                    continue

                bass_pc = min(piano_notes) % 12
                chord_root = chord.split(":")[0]
                chord_root_pc = ROOTS.index(chord_root)

                # Only count when bass confirms the chord root
                if bass_pc != chord_root_pc:
                    continue

                for pc in melody_by_beat[t]:
                    counts[chord][str(pc % 12)] += 1

        except Exception:
            skipped += 1
            continue

    if skipped:
        print(f"  Skipped {skipped} songs in bass-filtered emission learning")

    return {
        chord: {pc: v / sum(row.values()) for pc, v in row.items()}
        for chord, row in counts.items()
    }


def combine_emissions(
    free_midi_probs: dict,
    pop909_probs: dict,
    pop909_weight: float = 0.7,
) -> dict:
    """Blend free-midi-chords and POP909 emission probabilities.

    Args:
        free_midi_probs: Emissions learned from chord voicings.
        pop909_probs:    Emissions learned from melody-over-chord data.
        pop909_weight:   Weight given to POP909 (melody) data (0–1).

    Returns:
        Combined emission dict.
    """
    free_weight = 1.0 - pop909_weight
    combined = {}
    all_chords = set(free_midi_probs) | set(pop909_probs)
    for chord in all_chords:
        free = free_midi_probs.get(chord, {})
        pop  = pop909_probs.get(chord, {})
        all_pcs = set(free) | set(pop)
        combined[chord] = {
            pc: pop909_weight * pop.get(pc, 0.0) + free_weight * free.get(pc, 0.0)
            for pc in all_pcs
        }
    return combined


def learn_emissions_from_nottingham(
    nottingham_midi_dir: str,
    smoothing: float = 0.3,
) -> dict:
    """Learn P(pitch_class | chord) from Nottingham melody-over-chord pairs.

    Parses all MIDI files in the Nottingham dataset, aligns melody notes with
    identified chord labels beat-by-beat, and builds a pitch-class histogram
    per chord — same format as learn_emissions_from_pop909().

    Args:
        nottingham_midi_dir: Path to the Nottingham MIDI/ directory.
        smoothing:           Laplace smoothing added to every pitch-class count.

    Returns:
        Dict {chord_label: {str(pitch_class): probability}}
    """
    from harmonizer.parse_nottingham import load_all_nottingham

    counts: dict[str, dict[str, float]] = {
        chord: {str(pc): smoothing for pc in range(12)}
        for chord in CHORD_VOCAB
        if chord != "N"
    }

    corpus = load_all_nottingham(nottingham_midi_dir)
    total_pairs = 0

    for melody_seq, chord_seq in corpus:
        for melody_notes, chord in zip(melody_seq, chord_seq):
            if not melody_notes or chord not in counts:
                continue
            for pc in melody_notes:
                counts[chord][str(pc % 12)] += 1
                total_pairs += 1

    print(f"  Learned emissions from {len(corpus)} Nottingham songs ({total_pairs:,} note-chord pairs)")

    return {
        chord: {pc: v / sum(row.values()) for pc, v in row.items()}
        for chord, row in counts.items()
    }


def save_emissions(probs: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(probs, f, indent=2)
    print(f"Emission probabilities saved to {path}")


def load_emissions(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
