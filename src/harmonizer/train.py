"""Parameter estimation — learn HMM parameters from a labeled corpus.

All estimation is done from scratch using maximum likelihood (count + smooth).
No external HMM training libraries are used.
"""

from __future__ import annotations
import numpy as np
from harmonizer.chord_vocab import CHORD_VOCAB
from harmonizer.hmm import HMM


def estimate_parameters(
    corpus: list[tuple[list[int], list[int]]],
    n_obs: int,
    smoothing: float = 1e-6,
) -> HMM:
    """Estimate HMM parameters from a labeled corpus via maximum likelihood.

    Args:
        corpus: List of (observation_sequence, state_sequence) pairs.
                Observations are pitch-class integers (0-11).
                States are chord indices into CHORD_VOCAB.
        n_obs: Size of the observation alphabet (12 for pitch classes).
        smoothing: Laplace smoothing constant.

    Returns:
        Trained HMM instance.
    """
    N = len(CHORD_VOCAB)
    M = n_obs

    pi_counts = np.full(N, smoothing)
    A_counts  = np.full((N, N), smoothing)
    B_counts  = np.full((N, M), smoothing)

    for obs_seq, state_seq in corpus:
        if not state_seq:
            continue
        pi_counts[state_seq[0]] += 1
        for t in range(1, len(state_seq)):
            A_counts[state_seq[t - 1], state_seq[t]] += 1
        for obs, state in zip(obs_seq, state_seq):
            B_counts[state, obs % M] += 1

    pi = pi_counts / pi_counts.sum()
    A  = A_counts  / A_counts.sum(axis=1, keepdims=True)
    B  = B_counts  / B_counts.sum(axis=1, keepdims=True)

    return HMM(pi=pi, A=A, B=B)
