from harmonizer.chord_vocab import get_chord_to_roman_map_for_key, get_roman_transitions_for_key


def transition_score(prev_chord: str, next_chord: str, key: str = "C_major") -> float:
    """Score how naturally one chord moves to another in the given key."""
    chord_to_roman = get_chord_to_roman_map_for_key(key)
    roman_transitions = get_roman_transitions_for_key(key)

    prev_roman = chord_to_roman[prev_chord]
    next_roman = chord_to_roman[next_chord]

    if next_roman in roman_transitions.get(prev_roman, []):
        return 0.70

    if prev_chord == next_chord:
        return 0.20

    return 0.05
