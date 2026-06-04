"""HMM inference — Viterbi (1-best) and k-best decoding, implemented from scratch.

No external HMM libraries are used. All matrix operations use numpy only.
"""

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
        self.N = A.shape[0]  # number of hidden states
        self.M = B.shape[1]  # observation vocabulary size

    def viterbi_harmonize(melody_by_measure: list[list[int]], key: str = "C_major") -> list[str]:
        """Predict one chord per measure using a simple HMM/Viterbi algorithm."""
        states, scale_pcs = get_chords_for_key(key)

        if not melody_by_measure:
            return []

        dp: list[dict[str, float]] = []
        backpointer: list[dict[str, str | None]] = []

        first_scores: dict[str, float] = {}
        first_backpointers: dict[str, str | None] = {}

        for state in states:
            first_scores[state] = emission_score(melody_by_measure[0], state, scale_pcs)
            first_backpointers[state] = None

        dp.append(first_scores)
        backpointer.append(first_backpointers)

        for time_step in range(1, len(melody_by_measure)):
            current_scores: dict[str, float] = {}
            current_backpointers: dict[str, str | None] = {}

            for current_state in states:
                best_score = -1.0
                best_previous_state: str | None = None
                current_emission = emission_score(melody_by_measure[time_step], current_state, scale_pcs)

                for previous_state in states:
                    candidate_score = (
                        dp[time_step - 1][previous_state]
                        * transition_score(previous_state, current_state, key)
                        * current_emission
                    )

                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_previous_state = previous_state

                current_scores[current_state] = best_score
                current_backpointers[current_state] = best_previous_state

            dp.append(current_scores)
            backpointer.append(current_backpointers)

        best_last_state = max(dp[-1], key=dp[-1].get)
        best_path = [best_last_state]

        for time_step in range(len(melody_by_measure) - 1, 0, -1):
            previous_state = backpointer[time_step][best_path[-1]]
            if previous_state is None:
                break
            best_path.append(previous_state)

        best_path.reverse()
        return best_path

    def k_best_viterbi(self, observations: list[int], k: int) -> list[tuple[float, list[int]]]:
        """Return the k most likely state sequences (lazy beam approximation).

        Args:
            observations: List of integer observation indices.
            k: Number of best paths to return.

        Returns:
            List of (log_probability, state_sequence) tuples, best-first.
        """
        raise NotImplementedError
