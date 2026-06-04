from harmonizer.hmm import HMM
from harmonizer.chord_vocab import get_chord_to_roman_map_for_key

def demo_c_major_hmm() -> list[str]:
    """Run a tiny C major smoke test for the temporary HMM logic."""
    # melody_by_measure = [
    #     [0, 4, 7],      # C E G
    #     [5, 9, 0],      # F A C
    #     [7, 11, 2],     # G B D
    #     [0, 4, 7],      # C E G
    # ]
    melody_by_measure = [

    [2, 5, 9],      # D F A

    [7, 11, 2],     # G B D

    [0, 4, 7],      # C E G

    ]   # C:maj -> A:min -> F:maj -> C:maj

    predicted_chords = HMM.viterbi_harmonize(melody_by_measure, key="C_major")
    chord_to_roman = get_chord_to_roman_map_for_key("C_major")

    print("Predicted chords:")
    print(" -> ".join(predicted_chords))

    print("Roman numerals:")
    print(" -> ".join(chord_to_roman[chord] for chord in predicted_chords))

    return predicted_chords


if __name__ == "__main__":
    demo_c_major_hmm()