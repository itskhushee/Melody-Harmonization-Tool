# Melody-Harmonization Tool

## Problem
Given a monophonic melody (sequence of notes), automatically generate a harmonically plausible chord progression using a Hidden Markov Model (HMM). The melody notes are observations; chord labels are hidden states. The system learns transition and emission probabilities from a labeled corpus and uses the Viterbi algorithm to find the most likely chord sequence.

## Setup
```bash
# Clone the repo
git clone <repo-url>
cd Melody-Harmonization-Tool

# Install dependencies (Python 3.10+)
pip install -e .
# or
pip install -r requirements.txt
```

## Reproduction
```bash
# Train the HMM on a labeled MIDI corpus
python -m harmonizer.cli train --data data/processed/train.json --out models/hmm.pkl

# Harmonize a melody MIDI file
python -m harmonizer.cli harmonize --model models/hmm.pkl --input data/raw/melody.mid --output output/result.mid

# Run all tests
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
├── src/harmonizer/
│   ├── __init__.py
│   ├── midi_io.py        # MIDI parsing → note observations
│   ├── chord_vocab.py    # chord definitions (hidden states)
│   ├── hmm.py            # Viterbi + k-best (from scratch)
│   ├── train.py          # parameter estimation (from scratch)
│   ├── harmonize.py      # end-to-end pipeline
│   └── cli.py            # python -m harmonizer.cli ...
├── tests/
├── data/
│   ├── raw/              # source MIDI files (gitignored)
│   └── processed/        # cleaned, parsed datasets
├── notebooks/            # preprocessing / EDA only
└── docs/
    ├── project_summary.md
    ├── progress_log.md
    └── ai_use_disclosure.md
```