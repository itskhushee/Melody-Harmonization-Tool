from harmonizer.chord_vocab import chord_pitch_classes


def emission_score(
    melody_notes: list[int],
    chord_label: str,
    scale_pcs: set[int] | None = None,
    learned_emissions: dict | None = None,
) -> float:
    """Score how well a measure's melody notes fit a chord.

    Args:
        melody_notes:      Melody notes as pitch classes (C=0 … B=11).
        chord_label:       Internal chord label, e.g. "C:maj" or "B:dim".
        scale_pcs:         Pitch classes belonging to the current key (theory fallback).
        learned_emissions: If provided, use data-driven P(pitch_class | chord)
                           from train_emissions.learn_emissions(). Falls back to
                           music-theory weights when None.

    Returns:
        A positive score. Higher means the melody notes fit the chord better.
    """
    if not melody_notes:
        return 1e-6

    if learned_emissions is not None:
        chord_probs = learned_emissions.get(chord_label, {})
        if not chord_probs:
            return 1e-6
        total = sum(chord_probs.get(str(note % 12), 1e-6) for note in melody_notes)
        return total / len(melody_notes)

    # Theory-based fallback
    chord_tones = chord_pitch_classes(chord_label)
    if not chord_tones:
        return 1e-6

    score = 0.0
    for note in melody_notes:
        pitch_class = note % 12
        if pitch_class in chord_tones:
            score += 1.0
        elif scale_pcs is not None and pitch_class in scale_pcs:
            score += 0.25
        else:
            score += 0.05

    return score / len(melody_notes)
