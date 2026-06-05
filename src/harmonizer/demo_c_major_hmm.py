from __future__ import annotations

"""Run the C-major HMM on a real MIDI file.

Example:
    PYTHONPATH=src python -m harmonizer.demo_c_major_midi data/c_major_melodies/example.mid
"""



import json
import sys
import time
from pathlib import Path

from harmonizer.chord_vocab import get_chord_to_roman_map_for_key
from harmonizer.hmm import HMM
from harmonizer.midi_parser import midi_to_melody_by_measure

_DEBUG_LOG = Path(__file__).resolve().parents[2] / ".cursor" / "debug-6c13a9.log"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # region agent log
    payload = {
        "sessionId": "6c13a9",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with _DEBUG_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    # endregion


def _resolve_midi_path(argv_path: Path) -> Path:
    """Resolve MIDI path: absolute as-is, else cwd, else repo data/ prefix."""
    if argv_path.is_absolute() and argv_path.exists():
        return argv_path
    candidates = [
        argv_path,
        Path.cwd() / argv_path,
        _DEBUG_LOG.parents[1] / "data" / argv_path,
        _DEBUG_LOG.parents[1] / argv_path,
    ]
    if not str(argv_path).startswith("data/"):
        candidates.append(_DEBUG_LOG.parents[1] / "data" / argv_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return argv_path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  PYTHONPATH=src python -m harmonizer.demo_c_major_midi path/to/file.mid")
        return

    argv_path = Path(sys.argv[1])
    repo_root = _DEBUG_LOG.parents[1]
    candidates = [
        argv_path,
        Path.cwd() / argv_path,
        repo_root / "data" / argv_path,
        repo_root / argv_path,
    ]
    if not str(argv_path).startswith("data/"):
        candidates.append(repo_root / "data" / argv_path)
    # region agent log
    _debug_log(
        "A",
        "demo_c_major_hmm.py:main",
        "path resolution inputs",
        {
            "argv": sys.argv[1],
            "cwd": str(Path.cwd()),
            "repo_root": str(repo_root),
            "candidate_exists": {str(c): c.exists() for c in candidates},
        },
    )
    # endregion
    midi_path = _resolve_midi_path(argv_path)
    # region agent log
    _debug_log(
        "B",
        "demo_c_major_hmm.py:main",
        "resolved midi path",
        {"midi_path": str(midi_path), "exists": midi_path.exists()},
    )
    # endregion
    if not midi_path.exists():
        print(f"Error: MIDI file not found: {argv_path}")
        print(f"  cwd: {Path.cwd()}")
        print("  Try: data/raw/demo/train/c_major_pop.mid")
        return

    melody_by_measure = midi_to_melody_by_measure(midi_path)

    print(f"Loaded MIDI: {midi_path}")
    print("\nMelody by measure:")
    for i, measure_notes in enumerate(melody_by_measure, start=1):
        print(f"Measure {i}: {measure_notes}")

    predicted_chords = HMM.viterbi_harmonize(melody_by_measure, key="C_major")
    chord_to_roman = get_chord_to_roman_map_for_key("C_major")

    print("\nPredicted chords:")
    print(" -> ".join(predicted_chords))

    print("\nRoman numerals:")
    print(" -> ".join(chord_to_roman[chord] for chord in predicted_chords))


if __name__ == "__main__":
    main()