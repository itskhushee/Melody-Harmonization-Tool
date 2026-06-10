# Melody-Harmonization Tool

## Problem

Given a monophonic melody (sequence of notes), automatically generate a harmonically plausible chord progression using a Hidden Markov Model (HMM). Melody notes are observations; chord labels are hidden states. The system learns transition and emission probabilities from a multi-source labeled corpus (POP909, Nottingham, free-midi-chords) and uses the log-space Viterbi algorithm to find the most likely chord sequence.

## Setup

```bash
# Clone the repo (with submodules for datasets)
git clone --recurse-submodules <repo-url>
cd Melody-Harmonization-Tool

# Install dependencies (Python 3.10+)
pip install -e .
# or
pip install -r requirements.txt
```

> **Datasets** are included as git submodules under `data/raw/`:
> - `POP909-Dataset` — 909 annotated pop songs (train/test split 88/12)
> - `free-midi-chords` — chord voicing MIDI files across all 12 keys
> - `nottingham-dataset` — 1,034 folk melody + chord pairs

## Reproducing Results

### 1. Train the model

```bash
python3 src/harmonizer/cli.py train \
  --pop909      data/raw/POP909-Dataset/POP909 \
  --midi-chords data/raw/free-midi-chords/midi_files \
  --nottingham  data/raw/nottingham-dataset/MIDI \
  --out         models/hmm.json
```

### 2. Evaluate accuracy on the POP909 test set

```bash
python3 src/harmonizer/cli.py evaluate \
  --model       models/hmm.json \
  --pop909      data/raw/POP909-Dataset/POP909 \
  --granularity beat
```

Expected output: `Accuracy: ~10706/34272 = 31.2%`

### 3. Harmonize a melody MIDI file

```bash
python3 src/harmonizer/cli.py harmonize \
  --model  models/hmm.json \
  --input  data/raw/demo/train/ode_joy_c_major.mid \
  --output output.mid \
  --key    C_major
```

The output MIDI contains two tracks: the original melody and the predicted chord progression.

### 4. Run tests

```bash
pytest tests/
```

## Repository Layout

```
Melody-Harmonization-Tool/
├── README.md
├── CONTRIBUTING.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── models/
│   ├── hmm.json              # trained model (pi, A, B matrices)
│   ├── emissions.json        # standalone emission probabilities
│   └── transitions.json      # standalone transition probabilities
├── src/harmonizer/
│   ├── chord_vocab.py        # 37-state chord vocabulary, Roman numeral mappings
│   ├── hmm.py                # log-space Viterbi decoder (from scratch)
│   ├── harmonize.py          # end-to-end pipeline: MIDI in → chords → MIDI out
│   ├── midi_parser.py        # MIDI → beat/measure note sequences
│   ├── midi_io.py            # harmonized MIDI output writer
│   ├── train_transitions.py  # learn pi and A from POP909 beat-level bigrams
│   ├── train_emissions.py    # learn B from POP909, free-midi-chords, Nottingham
│   ├── parse_nottingham.py   # Nottingham MIDI parser (Jaccard chord identification)
│   ├── evaluate.py           # beat/measure accuracy against POP909 ground truth
│   └── cli.py                # train / harmonize / evaluate subcommands
├── tests/
│   └── test_c_major_hmm.py   # 6 pytest unit tests
├── data/
│   └── raw/
│       ├── POP909-Dataset/     # git submodule
│       ├── free-midi-chords/   # git submodule
│       ├── nottingham-dataset/ # git submodule
│       └── demo/               # short demo MIDI files for quick testing
└── docs/
    ├── project_summary.md
    ├── progress_log.md
    └── ai_use_disclosure.md
```
