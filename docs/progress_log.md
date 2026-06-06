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
- Model saved to `models/hmm.json` (14 KB).
- Fixed bugs: `demo_c_major_hmm.py` classmethod call, broken `cli.py` imports.
- Accuracy improvements applied: joint log-emission (sum not average), time-weighted key detection, MELODY-track priority, smoothing 1.0→0.1.

<!-- Add entries as work progresses -->
