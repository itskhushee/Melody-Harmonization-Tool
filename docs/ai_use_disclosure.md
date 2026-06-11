# AI Use Disclosure

This document logs all AI tool use on the Melody-Harmonization Tool project, per course requirements.

---

## Format

Each entry follows this structure:

| Field | Description |
|-------|-------------|
| **Date** | When the AI was used |
| **Tool** | Name of the AI tool |
| **Task** | What you were trying to accomplish |
| **Prompt / Interaction** | Summary of what was asked |
| **Output Used** | What output was incorporated into the project, and where |
| **Modifications** | Changes made to the AI output before using it |
| **Contributor** | Team member who used the tool |

---

## Log

---

### Entry 1

- **Date:** 2026-05-27
- **Tool:** Claude Code (claude-sonnet-4-6)
- **Task:** Project scaffolding — create `docs/ai_use_disclosure.md`
- **Prompt / Interaction:** *"Add `ai_use_disclosure.md` to my Melody-Harmonization-Tool into my Foundations of AI folder on my Desktop"*
- **Output Used:** This file (`docs/ai_use_disclosure.md`) — initial template and structure
- **Modifications:** None; used as generated
- **Contributor:** Alessandro Lancia

---

### Entry 2

- **Date:** 2026-06-04
- **Tool:** Claude Code (claude-sonnet-4-6)
- **Task:** Build core HMM with hardcoded C major values and verify Viterbi decoding on synthetic data
- **Prompt / Interaction:** Asked for help implementing the HMM skeleton — hardcoded pi, A, B matrices for C major, running Viterbi on fake note sequences to confirm decoding was correct before connecting real training data
- **Output Used:**
  - `hmm.py` — initial HMM class skeleton with hardcoded C major pi, A, B and basic Viterbi loop
  - `tests/test_c_major_hmm.py` — initial synthetic test cases to verify Viterbi traceback
- **Modifications:** Manually verified Viterbi output on paper against expected best-path; adjusted matrix values to confirm edge cases
- **Contributor:** Khushee Vakil

---

### Entry 3

- **Date:** 2026-06-05
- **Tool:** Claude Code (claude-sonnet-4-6)
- **Task:** Build full training pipeline and end-to-end harmonization pipeline
- **Prompt / Interaction:** Extended session covering:
  - MIDI parser to extract monophonic melody by beat and by measure from POP909 files
  - Generalizing HMM from hardcoded C major to all 24 keys programmatically
  - Learning transition probabilities (pi, A) from POP909 chord annotations
  - Learning emission probabilities (B) from free-midi-chords MIDI voicings
  - Connecting pi, A, B to the HMM class with log-space Viterbi and full traceback
  - End-to-end pipeline: MIDI in → note parsing → Viterbi → predicted chords → two-track MIDI out
  - Adding `_smooth_predictions()` to remove single-beat A→B→A blips
  - CLI polish: measure-level MIDI output for playback, beat-level evaluation for accuracy
- **Output Used:**
  - `midi_parser.py` — beat/measure MIDI parsing, melody track selection heuristic
  - `chord_vocab.py` — 24-key generalization, Roman numeral mappings, key detection
  - `train_transitions.py` — Roman-numeral bigram probability learning from POP909
  - `train_emissions.py` — P(pitch_class | chord) learning from free-midi-chords
  - `hmm.py` — log-space Viterbi with real pi/A/B, `_smooth_predictions()`
  - `harmonize.py` — full MIDI-in to MIDI-out pipeline
  - `cli.py` — train and harmonize subcommands
- **Modifications:** Reviewed all generated code; tested pipeline end-to-end on demo MIDIs before committing; adjusted melody track priority logic and smoothing window manually
- **Contributor:** Khushee Vakil

---

### Entry 4

- **Date:** 2026-06-06
- **Tool:** Claude Code (claude-sonnet-4-6)
- **Task:** Improve Viterbi decoder and training pipeline — accuracy improvements before first evaluation run
- **Prompt / Interaction:** Asked for help improving emission discriminability (joint log-probability vs averaging), fixing smoothing parameter default, and improving the Viterbi boost logic
- **Output Used:**
  - `hmm.py` — `log_obs` changed to sum (not average) of log-emissions; chord-tone boost logic refined; `_smooth_predictions` loop corrected
  - `train_transitions.py` — smoothing default lowered from 1.0 to 0.1
  - `models/hmm.json` — first trained model committed
- **Modifications:** Verified accuracy improvement before committing; retained own implementations of Viterbi DP and traceback
- **Contributor:** Khushee Vakil

---

### Entry 5

- **Date:** 2026-06-06
- **Tool:** Claude Code (claude-sonnet-4-6)
- **Task:** Bug fixes, CLI rewrite, accuracy improvements, Nottingham integration, MIDI output quality
- **Prompt / Interaction:** Series of prompts to fix broken CLI (references to deleted `train.py`), fix silent pi learning bug, switch transition learning to beat-level bigrams, tune chord-tone boost via grid search, integrate Nottingham emissions, and improve MIDI output compression
- **Output Used:**
  - `cli.py` — full rewrite with `evaluate` subcommand
  - `train_transitions.py` — beat-level bigram counting, pi learning fix
  - `parse_nottingham.py` — new file (Jaccard chord identification from MIDI triads)
  - `train_emissions.py` — `learn_emissions_from_nottingham()` and 3-way blend
  - `hmm.py` — `chord_tone_boost` as tunable parameter, `log_obs` sum not average
  - `midi_io.py` — `_compress_chords()` for MIDI output quality
  - `tests/test_c_major_hmm.py` — 6 proper pytest unit tests (replaced broken demo)
  - `models/hmm.json` — retrained model achieving 31.2% beat-level accuracy
- **Modifications:** Reviewed all diffs; beat-level transition logic and Nottingham blend ratios verified against accuracy results before committing
- **Contributor:** Alessandro Lancia

---

### Entry 6

- **Date:** 2026-06-09
- **Tool:** Claude Code (claude-sonnet-4-6)
- **Task:** Project audit, bug diagnosis, transition analysis, report preparation
- **Prompt / Interaction:** Extended session covering:
  - Full project status audit (what is done, current accuracy)
  - Diagnosing the all-C:maj harmonization bug — traced root cause to interaction of π[I]=73.9%, A[I,I]=0.781, and `_smooth_predictions` collapsing alternating patterns
  - Tracing which commit introduced the bug (teammate's `d65e122` accuracy fixes)
  - Computing transition statistics from POP909 training data (78.1% I→I self-transition)
  - Generating `transition_analysis.png` (bar chart + A matrix heatmap)
  - Testing on long POP909 and Nottingham songs to validate report arguments
- **Contributor:** Khushee Vakil

---

<!-- Add new entries below as you use AI tools throughout the project -->
