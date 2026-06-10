# Progress Log

Narrative supplement to the GitHub Project board.

---

## 2026-05-27

- Initialized repository structure: all source modules, tests scaffold, data directories, and docs.
- Added project scaffold: `src/harmonizer/`, `tests/`, `data/`, `docs/`, `pyproject.toml`, `requirements.txt`.

---

## 2026-06-01

- Added demo MIDI corpus under `data/raw/demo/` (Twinkle, Ode to Joy, Mary Had a Little Lamb, C major scale/arpeggio, C minor lament).
- Updated `.gitignore` to track raw data submodules.

---

## 2026-06-03

- Added POP909 and Nottingham datasets as git submodules under `data/raw/`.
- Expanded chord vocabulary to all 12 keys × major/minor triads (later simplified to 37 states: 12 roots × 3 qualities + no-chord).

---

## 2026-06-04

- Got core HMM working with hardcoded C major values on fake data — Viterbi decoding confirmed correct on synthetic test cases.

---

## 2026-06-05 (Khushee)

- **MIDI parser**: `midi_parser.py` — extracts monophonic melody by beat and by measure, prefers track named `melody` before falling back to highest-pitch heuristic.
- **Chord vocabulary**: generalized from hardcoded C major to all 24 keys programmatically via `chord_vocab.py`.
- **Transition learning**: `train_transitions.py` — learned Roman-numeral bigram probabilities from POP909 chord annotations.
- **Emission learning**: `train_emissions.py` — learned P(pitch_class | chord) from free-midi-chords MIDI voicings.
- **HMM class**: connected pi, A, B matrices to log-space Viterbi in `hmm.py`; implemented full traceback.
- **End-to-end pipeline**: `harmonize.py` — MIDI in → note parsing → Viterbi → predicted chords → two-track MIDI out.
- **Smoothing**: added `_smooth_predictions()` to remove single-beat A→B→A blips from Viterbi output.
- **CLI polish**: measure-level MIDI output for natural playback, beat-level evaluation for accuracy metrics.

---

## 2026-06-06 (Alessandro)

- Fixed bugs: `demo_c_major_hmm.py` classmethod call, broken `cli.py` (referenced deleted `train.py`), rewrote 6 proper pytest tests.
- Added `evaluate` subcommand to CLI.
- Accuracy improvements:
  - Joint log-emission (sum not average in `log_obs`) — more discriminative signal on longer measures.
  - Time-weighted key detection in `chord_vocab.py` — picks dominant key by duration, not last token.
- **Baseline accuracy (109-song POP909 test set):**
  - Beat-level: **29.6%** (10,128 / 34,272)
  - Measure-level: **22.4%** (3,403 / 15,175)
  - Random baseline (1-of-7 diatonic chords): ~14.3% — model is ~2× better than random.

- **Three targeted accuracy fixes (beat 29.6% → 31.0%, +1.4pp):**
  1. **Beat-level A matrix**: `learn_transitions` now counts beat-by-beat bigrams from `beat_midi.txt`, so self-transitions are naturally represented. Old segment-level approach never counted I→I, causing A to actively penalise chord persistence. A_major diagonal is now ~0.65–0.78.
  2. **Fixed silent pi learning bug**: `learn_pi` was calling removed helper `_read_chord_sequence`, causing NameError caught by broad `except` → all songs skipped → pi stuck at uniform 1/7. Rewrote to scan first in-key beat: I=73.9%, IV=14.6%, vi=7.0%.
  3. **Tuning**: chord-tone boost 3.0→10.0 (grid search); POP909 emission weight 50%→70% (bass-confirmed beats only); smoothing corrected 1.0→0.1.

- **Nottingham integration (beat 31.0% → 31.2%, +0.2pp):**
  - Added `parse_nottingham.py`: identifies chords from Track 1 triad pitch-class sets via Jaccard similarity; extracts 138,540 melody-over-chord beat pairs from 1,034 songs.
  - Grid search: Nottingham transitions hurt (-2.4pp, folk grammar ≠ pop grammar), Nottingham emissions help (+0.2pp).
  - Optimal 3-way emission blend: **65% POP909 + 20% Nottingham + 15% free-midi-chords**.
  - **Final accuracy:**
    - Beat-level: **31.2%** (10,706 / 34,272) — +1.6pp vs original baseline
    - Measure-level: **23.7%** (3,595 / 15,175)

- **MIDI output quality**: `midi_io.py` — added `_compress_chords()` to merge consecutive identical chords, min 2 beats (absorbs flicker), max 8 beats (prevents monotony). `chord_tone_boost` now a tunable parameter (10.0 for accuracy, 4.0–5.0 for musical variety).

---

## 2026-06-09 (Khushee)

- **Diagnosed all-C:maj harmonization bug**: model was predicting only the tonic chord for all demo melodies regardless of input. Root cause identified as the interaction of three factors:
  1. π[I] = 73.9% — learned initial state heavily biased to tonic (correct for POP909 but dominates short melodies).
  2. A[I,I] = 0.781 — high self-transition keeps Viterbi on tonic once started.
  3. `_smooth_predictions` running in a `while changed` loop collapses valid alternating patterns (e.g. C:maj→D:min→C:maj→E:min) back to all-C:maj.
- Confirmed the bug was introduced by the teammate's 2026-06-06 accuracy fixes: the beat-level transition fix correctly raised A[I,I] from ~0.14 to 0.78, but combined with the pi fix and smoothing, made harmonization monotonic.
- **Transition analysis on POP909**: computed that 78.1% of beat-level transitions from I are self-transitions (I→I = 27,035 out of 34,636). Dataset chord distribution is actually balanced (I = 16.2%, V = 17.0%) — the bias is in the *learned parameters*, not the raw data.
- Generated `transition_analysis.png`: bar chart of I transitions + full 7×7 A matrix heatmap, for inclusion in the project report.
- Tested on long POP909 test song (song 197, 576 beats, C major): confirmed all-C:maj output even on long sequences, validating that the bug is parameter-driven not sequence-length-driven.
- Tested on Nottingham monophonic songs: `ashover37.mid` (238 beats, D major) produced `D:maj → B:min → A:maj` (I–vi–V progression), confirming the model produces varied output when melody notes provide sufficient emission signal over many beats.
- Ground-truth analysis of song 197: 47% of first 32 beats are chromatic chords (E:maj, A:maj, A#:maj) outside the 7-state diatonic vocabulary — identified as second reason for 31.2% accuracy ceiling.

---

<!-- Add entries as work progresses -->
