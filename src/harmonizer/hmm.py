"""HMM inference — Viterbi decoding implemented from scratch using numpy."""

from __future__ import annotations
import numpy as np
from harmonizer.chord_vocab import get_chords_for_key
from harmonizer.emissions import emission_score
from harmonizer.transitions import transition_score


class HMM:
    """Discrete Hidden Markov Model.

    Attributes:
        pi:  Initial state distribution, shape (N,)
        A:   Transition matrix, shape (N, N)  A[i,j] = P(s_t=j | s_{t-1}=i)
        B:   Emission matrix, shape (N, M)    B[i,k] = P(o_t=k | s_t=i)
    """

    def __init__(self, pi: np.ndarray, A: np.ndarray, B: np.ndarray) -> None:
        self.pi = pi
        self.A = A
        self.B = B
        self.N = A.shape[0]
        self.M = B.shape[1]

    @staticmethod
    def viterbi_harmonize(
        melody_by_measure: list[list[int]],
        key: str = "C_major",
        learned_probs: dict | None = None,
        learned_emissions: dict | None = None,
    ) -> list[str]:
        """Predict one chord per measure using the Viterbi algorithm.

        Args:
            melody_by_measure: List of measures; each measure is a list of
                               pitch classes (0–11).
            key:               Key to harmonize in, e.g. "G_major", "D_minor".
            learned_probs:     Optional data-driven transition probabilities from
                               train_transitions.learn_transitions(). Uses
                               music-theory weights when None.

        Returns:
            List of chord label strings, one per measure.
        """
        states, scale_pcs = get_chords_for_key(key)

        if not melody_by_measure:
            return []

        dp: list[dict[str, float]] = []
        backpointer: list[dict[str, str | None]] = []

        # Initialise with emission scores only (no prior transition)
        first_scores = {
            state: emission_score(melody_by_measure[0], state, scale_pcs, learned_emissions)
            for state in states
        }
        dp.append(first_scores)
        backpointer.append({state: None for state in states})

        for t in range(1, len(melody_by_measure)):
            current_scores: dict[str, float] = {}
            current_backpointers: dict[str, str | None] = {}
            current_emission = {
                state: emission_score(melody_by_measure[t], state, scale_pcs, learned_emissions)
                for state in states
            }

            for curr in states:
                best_score = -1.0
                best_prev: str | None = None

                for prev in states:
                    score = (
                        dp[t - 1][prev]
                        * transition_score(prev, curr, key, learned_probs)
                        * current_emission[curr]
                    )
                    if score > best_score:
                        best_score = score
                        best_prev = prev

                current_scores[curr] = best_score
                current_backpointers[curr] = best_prev

            dp.append(current_scores)
            backpointer.append(current_backpointers)

        # Traceback
        best_last = max(dp[-1], key=dp[-1].get)
        path = [best_last]
        for t in range(len(melody_by_measure) - 1, 0, -1):
            prev = backpointer[t][path[-1]]
            if prev is None:
                break
            path.append(prev)

        path.reverse()
        return path

    def k_best_viterbi(self, observations: list[int], k: int) -> list[tuple[float, list[int]]]:
        """Return the k most likely state sequences (lazy beam approximation)."""
        raise NotImplementedError
