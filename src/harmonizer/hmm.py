"""HMM — trained model with pi, A, B and log-space Viterbi decoding."""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from harmonizer.chord_vocab import get_chords_for_key


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
        log_B = np.log(B + 1e-10)            # shape (7, 12)

        # ── Helper: sum log-emissions for all notes in a measure ────────────
        # Sum = joint log-probability (notes i.i.d. given chord).
        # More discriminative than averaging: longer measures give more signal.
        def log_obs(t: int, state_idx: int) -> float:
            notes = melody_by_measure[t]
            if not notes:
                return 0.0
            return sum(log_B[state_idx, pc % 12] for pc in notes)

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

        return [states[i] for i in path]

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
