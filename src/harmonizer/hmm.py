"""HMM inference — Viterbi (1-best) and k-best decoding, implemented from scratch.

No external HMM libraries are used. All matrix operations use numpy only.
"""

from __future__ import annotations
import numpy as np


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

    def viterbi(self, observations: list[int]) -> list[int]:
        """Return the most likely state sequence for the observation sequence.

        Args:
            observations: List of integer observation indices.

        Returns:
            List of integer state indices (same length as observations).
        """
        T = len(observations)
        log_pi = np.log(self.pi + 1e-300)
        log_A  = np.log(self.A  + 1e-300)
        log_B  = np.log(self.B  + 1e-300)

        delta = np.full((T, self.N), -np.inf)
        psi   = np.zeros((T, self.N), dtype=int)

        delta[0] = log_pi + log_B[:, observations[0]]

        for t in range(1, T):
            for j in range(self.N):
                scores = delta[t - 1] + log_A[:, j]
                psi[t, j]   = np.argmax(scores)
                delta[t, j] = scores[psi[t, j]] + log_B[j, observations[t]]

        # Backtrack
        path = [int(np.argmax(delta[T - 1]))]
        for t in range(T - 1, 0, -1):
            path.append(psi[t, path[-1]])
        path.reverse()
        return path

    def k_best_viterbi(self, observations: list[int], k: int) -> list[tuple[float, list[int]]]:
        """Return the k most likely state sequences (lazy beam approximation).

        Args:
            observations: List of integer observation indices.
            k: Number of best paths to return.

        Returns:
            List of (log_probability, state_sequence) tuples, best-first.
        """
        raise NotImplementedError
