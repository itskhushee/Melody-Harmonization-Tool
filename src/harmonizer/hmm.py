"""HMM — trained model with pi, A, B and log-space Viterbi decoding."""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from harmonizer.chord_vocab import get_chords_for_key, chord_pitch_classes


class HMM:
    """Trained Hidden Markov Model for chord prediction.

    Attributes:
        pi_major: Initial state distribution for major keys, shape (7,)
        pi_minor: Initial state distribution for minor keys, shape (7,)
        A_major:  Transition matrix for major keys, shape (7, 7)
        A_minor:  Transition matrix for minor keys, shape (7, 7)
        emissions: Dict {chord_label: {pc_str: probability}} from free-midi-chords
    """

    def __init__(
        self,
        pi_major: np.ndarray,
        pi_minor: np.ndarray,
        A_major: np.ndarray,
        A_minor: np.ndarray,
        emissions: dict,
    ) -> None:
        self.pi_major  = pi_major
        self.pi_minor  = pi_minor
        self.A_major   = A_major
        self.A_minor   = A_minor
        self.emissions = emissions

    @staticmethod
    def _fill_empty_beats(melody: list[list[int]], window: int = 4) -> list[list[int]]:
        """Fill empty beats by borrowing from the nearest non-empty neighbor.

        A beat with no melody notes contributes uniform emission to Viterbi,
        effectively letting the transition prior dominate. Borrowing from the
        adjacent beat (sustaining note context) gives the decoder real signal.
        """
        n = len(melody)
        filled = []
        for i, beat in enumerate(melody):
            if beat:
                filled.append(beat)
                continue
            context: list[int] = []
            for offset in range(1, window + 1):
                if i - offset >= 0 and melody[i - offset]:
                    context = list(melody[i - offset])
                    break
            if not context:
                for offset in range(1, window + 1):
                    if i + offset < n and melody[i + offset]:
                        context = list(melody[i + offset])
                        break
            filled.append(context)
        return filled

    @staticmethod
    def _smooth_predictions(chords: list[str]) -> list[str]:
        """Remove passing-tone blips from Viterbi output.

        Replaces any run of 1 beat that is sandwiched between the same chord
        on both sides (A → B → A pattern). This removes artefacts caused by
        melody passing tones without collapsing neighbouring chord changes.
        Runs multiple passes until no more X→Y→X patterns remain.
        """
        result = list(chords)
        changed = True
        while changed:
            changed = False
            for i in range(1, len(result) - 1):
                if result[i] != result[i - 1] and result[i] == result[i - 1]:
                    pass  # never true — placeholder avoided
                if result[i - 1] == result[i + 1] and result[i] != result[i - 1]:
                    result[i] = result[i - 1]
                    changed = True
        return result

    def viterbi_harmonize(
        self,
        melody_by_measure: list[list[int]],
        key: str = "C_major",
    ) -> list[str]:
        """Predict one chord per measure using log-space Viterbi decoding.

        Args:
            melody_by_measure: List of measures; each measure is a list of
                               pitch classes (0–11).
            key:               Key to harmonize in, e.g. "G_major", "D_minor".

        Returns:
            List of chord label strings, one per measure.
        """
        melody_by_measure = self._fill_empty_beats(melody_by_measure)
        states, _ = get_chords_for_key(key)
        _, mode   = key.rsplit("_", 1)
        n = len(states)
        T = len(melody_by_measure)

        if T == 0:
            return []

        # ── Build log matrices for this key ──────────────────────────────────
        pi = self.pi_major if mode == "major" else self.pi_minor
        A  = self.A_major  if mode == "major" else self.A_minor

        log_pi = np.log(pi + 1e-10)          # shape (7,)
        log_A  = np.log(A  + 1e-10)          # shape (7, 7)

        # B matrix: P(pitch_class | chord), shape (7, 12)
        B = np.zeros((n, 12))
        for i, chord in enumerate(states):
            for pc_str, prob in self.emissions.get(chord, {}).items():
                B[i, int(pc_str)] = prob

        # Boost chord tones so that non-matching observations more strongly
        # rule out wrong chords. Diatonic chords share scale tones, so without
        # this boost the B rows are too similar and the transition prior wins.
        # Value of 3.0 comes from empirical grid search; higher values cause
        # excess chord changes by over-reacting to passing tones.
        _CHORD_TONE_BOOST = 3.0
        for i, chord in enumerate(states):
            for pc in chord_pitch_classes(chord):
                B[i, pc] *= _CHORD_TONE_BOOST
        B /= B.sum(axis=1, keepdims=True)

        log_B = np.log(B + 1e-10)            # shape (7, 12)

        # ── Helper: average log-emission over all notes in a beat ────────────
        def log_obs(t: int, state_idx: int) -> float:
            notes = melody_by_measure[t]
            if not notes:
                return 0.0
            return sum(log_B[state_idx, pc % 12] for pc in notes) / len(notes)

        # ── Viterbi DP ────────────────────────────────────────────────────────
        delta = np.full((T, n), -np.inf)
        psi   = np.zeros((T, n), dtype=int)

        # Initialisation
        for i in range(n):
            delta[0, i] = log_pi[i] + log_obs(0, i)

        # Recursion
        for t in range(1, T):
            for j in range(n):
                candidates = delta[t - 1] + log_A[:, j]  # shape (7,)
                psi[t, j]   = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + log_obs(t, j)

        # Traceback
        path = np.zeros(T, dtype=int)
        path[T - 1] = np.argmax(delta[T - 1])
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return self._smooth_predictions([states[i] for i in path])

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pi_major":  self.pi_major.tolist(),
            "pi_minor":  self.pi_minor.tolist(),
            "A_major":   self.A_major.tolist(),
            "A_minor":   self.A_minor.tolist(),
            "emissions": self.emissions,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"HMM saved to {path}")

    @classmethod
    def load(cls, path: str) -> "HMM":
        with open(path) as f:
            data = json.load(f)
        return cls(
            pi_major  = np.array(data["pi_major"]),
            pi_minor  = np.array(data["pi_minor"]),
            A_major   = np.array(data["A_major"]),
            A_minor   = np.array(data["A_minor"]),
            emissions = data["emissions"],
        )
