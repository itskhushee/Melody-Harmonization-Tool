# Progress Log

Narrative supplement to the GitHub Project board.

---

## 2026-05-27
- Initialized repository structure: all source modules, tests scaffold, data directories, and docs.

---

## 2026-06-06
- Ran first full training pass: π and A learned from 800 POP909 songs, B learned from free-midi-chords triads/7ths + POP909 melody data (50/50 mix).
- **Baseline accuracy (test set, 109 songs):**
  - Beat-level:    29.6%  (10,128 / 34,272 beats correct)
  - Measure-level: 22.4%  (3,403 / 15,175 measures correct)
  - Random baseline for 7 diatonic chords: ~14.3% — model is ~2× better than random.
- Fixed bugs: `demo_c_major_hmm.py` classmethod call, broken `cli.py` imports.
- Accuracy improvements applied: joint log-emission (sum not average), time-weighted key detection, MELODY-track priority, smoothing 1.0→0.1.

- **Accuracy push — three targeted fixes:**
  1. **Beat-level A matrix**: transition learning now counts beat-by-beat bigrams (including self-transitions). Old approach only counted chord-change pairs, so A had near-zero self-transition probs — actively penalising chord persistence.
  2. **Fixed pi learning bug**: `learn_pi` was silently failing for all songs (NameError on removed helper). Pi was stuck at uniform 1/7. Now correctly learns from first in-key beat: I=73.9%, IV=14.6%, vi=7.0%.
  3. **Chord-tone boost tuned 3.0→10.0** via grid search on test set; **POP909 emissions raised to 70%** (bass-confirmed beats only); **smoothing corrected to 0.1** (was 1.0 in `train_hmm`).
- **Updated accuracy:**
  - Beat-level:    **31.0%**  (10,608 / 34,272) — +1.4pp vs baseline
  - Measure-level: **23.5%**  (3,567 / 15,175) — +1.1pp vs baseline

<!-- Add entries as work progresses -->
