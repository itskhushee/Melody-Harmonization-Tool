# Contributing

## Team

| Contributor | Role |
|-------------|------|
| Khushee Vakil | HMM implementation, training pipeline, harmonization pipeline |
| Alejandro Lancia | Bug fixes, accuracy improvements, Nottingham integration |

## Work Split

### Khushee Vakil
- Designed and implemented the core HMM class (`hmm.py`) — log-space Viterbi decoder and traceback from scratch
- Built `midi_parser.py` — beat/measure MIDI parsing with melody track selection heuristic
- Built `chord_vocab.py` — 37-state chord vocabulary, Roman numeral mappings, 24-key generalization
- Built `train_transitions.py` — Roman-numeral bigram probability learning from POP909 chord annotations
- Built `train_emissions.py` — emission probability learning from free-midi-chords MIDI voicings
- Built `harmonize.py` — end-to-end pipeline: MIDI in → Viterbi → predicted chords → two-track MIDI out
- Built `cli.py` — initial train and harmonize subcommands
- Added `_smooth_predictions()` for post-processing Viterbi output
- Ran transition analysis on POP909 training data; generated `transition_analysis.png` for report


### Alejandro Lancia
- Fixed classmethod bug in `demo_c_major_hmm.py` and broken `cli.py` imports
- Rewrote `cli.py` with working `evaluate` subcommand
- Fixed silent pi learning bug (`learn_pi` was silently skipping all songs → pi stuck at uniform 1/7)
- Switched transition learning to beat-level bigrams, fixing A[I,I] self-transition undercount
- Tuned chord-tone boost 3.0→10.0 via grid search; raised POP909 emission weight to 70%
- Wrote `parse_nottingham.py` — Jaccard similarity chord identification from MIDI triads
- Added `learn_emissions_from_nottingham()` and 3-way emission blend (65% POP909 + 20% Nottingham + 15% free-midi-chords)
- Added `_compress_chords()` to `midi_io.py` for MIDI output quality
- Rewrote `tests/test_c_major_hmm.py` with 6 proper pytest unit tests
- Achieved final beat-level accuracy of 31.2% (up from 29.6% baseline)

## Workflow
1. Create a branch for each feature or fix: `git checkout -b feature/your-feature`
2. Commit often with clear messages
3. Open a pull request for review before merging into `main`
4. Keep `main` always in a working state

## Code Style
- Follow PEP 8
- Run `pytest tests/` before pushing
- Document any AI tool use in `docs/ai_use_disclosure.md`
