from harmonizer.chord_vocab import get_chord_to_roman_map_for_key, get_roman_transitions_for_key


def transition_score(
    prev_chord: str,
    next_chord: str,
    key: str = "C_major",
    learned_probs: dict | None = None,
) -> float:
    """Score how naturally one chord moves to another in the given key.

    Args:
        prev_chord:    Chord label of the previous state, e.g. "G:maj".
        next_chord:    Chord label of the current state.
        key:           Key string, e.g. "C_major", "D_minor".
        learned_probs: If provided, use data-driven probabilities from
                       train_transitions.learn_transitions(). Falls back to
                       music-theory weights when None.

    Returns:
        A positive score. Higher means the transition is more likely.
    """
    chord_to_roman = get_chord_to_roman_map_for_key(key)
    prev_roman = chord_to_roman[prev_chord]
    next_roman = chord_to_roman[next_chord]

    if learned_probs is not None:
        _, mode = key.rsplit("_", 1)
        mode_probs = learned_probs.get(mode, {})
        return mode_probs.get(prev_roman, {}).get(next_roman, 1e-6)

    # Theory-based fallback
    roman_transitions = get_roman_transitions_for_key(key)
    if next_roman in roman_transitions.get(prev_roman, []):
        return 0.70
    if prev_chord == next_chord:
        return 0.20
    return 0.05
