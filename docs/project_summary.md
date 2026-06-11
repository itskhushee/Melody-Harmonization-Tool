# Project Summary

## Problem Statement

Given a monophonic melody (a sequence of single notes), automatically generate a harmonically plausible chord progression. This is the classic accompaniment problem in computational music: the melody is observable, but the underlying harmonic structure is hidden and must be inferred.

## Approach

We model the task as a Hidden Markov Model (HMM):
- **Hidden states** — diatonic chord labels (7 chords per key × 24 keys = 168 possible states, but restricted to the 7 diatonic chords of the detected key at inference time)
- **Observations** — pitch classes (0–11) present in each beat of the melody
- **Decoding** — log-space Viterbi algorithm recovers the most likely chord sequence

Parameters are estimated from two corpora:
- **π and A** (initial and transition distributions) — learned from POP909 chord annotations using key-agnostic Roman numeral bigrams (e.g., I→V, vi→IV), so patterns generalise across all keys
- **B** (emission probabilities) — blended from three sources: POP909 melody-over-chord pairs filtered by bass confirmation (65%), Nottingham folk MIDI melody-over-chord pairs (20%), and free-midi-chords voicing files (15%)

## Model Description

| Component | Detail |
|-----------|--------|
| States | 7 diatonic chords per key (maj, min, dim) |
| Observations | Pitch-class sets per beat (0–11) |
| Transition A | Roman-numeral bigrams from 800 POP909 training songs |
| Emission B | 3-way blended histogram (65% POP909 / 20% Nottingham / 15% free-midi-chords) |
| Decoding | Log-space Viterbi with chord-tone boost and empty-beat fill |
| Post-processing | Smooth single-beat blips; compress consecutive identical chords (min 2 beats, max 8 beats) |

**Training data:**
- POP909: 909 annotated pop songs (800 train / 109 test, fixed seed)
- Nottingham: 1,034 folk MIDI files — 138,540 beats, 190,193 note-chord pairs
- free-midi-chords: chord voicing MIDI files across all 12 keys

## Results

| Metric | Score |
|--------|-------|
| Beat-level accuracy (POP909 test) | **31.2%** |
| Measure-level accuracy (POP909 test) | **23.7%** |
| Random baseline (1-of-7 diatonic) | 14.3% |

The model is approximately **2.2× better than random**.

### Qualitative analysis — chord transitions on longer melodies

A key finding emerged from listening tests on Nottingham folk songs of varying length:

**Short melodies (< 32 beats):** The model tends to predict the tonic chord throughout. With few melody notes, the emission signal is too weak to overcome the transition penalty, and the model defaults to the highest-prior state (I).

**Longer melodies (> 100 beats):** The model produces varied, musically correct progressions. For example:

- **`reelsr-t65.mid` (281 beats, D major):** Mostly D:maj with recurring A:maj — I and V, which are the two most common chords in a major key.
- **`ashover37.mid` (238 beats, D major):** D:maj → B:min → A:maj repeating — this is **I → vi → V** in D major, a genuine and musically correct diatonic progression.
- **`waltzes7.mid` (193 beats, G major):** D:maj (V) dominates rather than G:maj (I) — the melody starts on D notes, which fit both I and V in G major, causing the model to settle on V. This is a plausible but non-tonic starting chord.

This directly demonstrates that the model does learn real harmonic structure:

> *"On longer folk melodies (193–281 beats), the model produces varied diatonic progressions such as I–vi–V (D:maj → B:min → A:maj in `ashover37.mid`, 238 beats), demonstrating that monotonic tonic prediction on short sequences is a consequence of insufficient emission signal to overcome the transition penalty — not a fundamental inability of the model to predict chord changes."*

## Limitations & Future Work

**Current limitations:**
- Restricted to 7 diatonic chords per key — cannot predict borrowed chords, secondary dominants, or chromatic chords that appear frequently in pop music (~40% of POP909 annotations are out-of-key)
- Short melodies (< 32 beats) produce tonic-only output due to weak emission signal
- Key must be provided at inference time; automatic key detection from melody is not yet implemented
- Emission model does not weight notes by duration or metrical position (strong vs. weak beats)

**Future work:**
- Add secondary dominants (V/V, V/vi) to the state space to handle chromatic chords
- Downbeat weighting: scale emission scores by beat position (beat 1 carries more harmonic information than beats 2–4)
- Automatic key detection from the melody itself using a Krumhansl–Schmuckler key-finding algorithm
- Use Nottingham chord sequences for transition training within a separate folk-music mode
