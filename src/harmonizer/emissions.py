from harmonizer.chord_vocab import chord_pitch_classes

def emission_score(melody_notes: list[int], chord_label: str, scale_pcs: set[int] | None = None) -> float:
    """Score how well a measure's melody notes fit a chord.

    Args:
        melody_notes: Melody notes as pitch classes, where C=0, C#=1, ..., B=11.
        chord_label: Internal chord label, such as "C:maj" or "B:dim".
        scale_pcs: Optional pitch classes belonging to the current key.

    Returns:
        A positive score. Higher means the melody notes fit the chord better.
    """
    if not melody_notes:
        return 0.01

    chord_tones = chord_pitch_classes(chord_label)
    if not chord_tones:
        return 0.01

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
