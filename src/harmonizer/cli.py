"""Command-line interface — python -m harmonizer.cli <subcommand> ..."""

from __future__ import annotations
import argparse
import json
import sys


def cmd_train(args: argparse.Namespace) -> None:
    from harmonizer.train_transitions import train_hmm, get_train_test_split

    train_ids, _ = get_train_test_split(args.pop909)
    hmm = train_hmm(
        pop909_dir    = args.pop909,
        train_ids     = train_ids,
        midi_files_dir= args.midi_chords,
    )
    hmm.save(args.out)


def cmd_harmonize(args: argparse.Namespace) -> None:
    from harmonizer.hmm import HMM
    from harmonizer.harmonize import harmonize_midi

    hmm    = HMM.load(args.model)
    chords = harmonize_midi(args.input, args.output, key=args.key, hmm=hmm)
    print(" ".join(chords))


def cmd_evaluate(args: argparse.Namespace) -> None:
    from harmonizer.hmm import HMM
    from harmonizer.train_transitions import get_train_test_split
    from harmonizer.evaluate import evaluate

    hmm = HMM.load(args.model)
    _, test_ids = get_train_test_split(args.pop909)
    results = evaluate(
        pop909_dir  = args.pop909,
        test_ids    = test_ids,
        hmm         = hmm,
        granularity = args.granularity,
        verbose     = args.verbose,
    )
    print(f"Accuracy: {results['correct']}/{results['total']} "
          f"= {results['accuracy']*100:.1f}%  "
          f"(skipped {results['skipped']} songs)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="harmonizer")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train HMM from POP909 + free-midi-chords")
    p_train.add_argument("--pop909",      required=True, help="Path to POP909/POP909/ directory")
    p_train.add_argument("--midi-chords", required=True, help="Path to free-midi-chords midi_files/ directory")
    p_train.add_argument("--out",         required=True, help="Output model .json path")

    # ── harmonize ────────────────────────────────────────────────────────────
    p_harm = sub.add_parser("harmonize", help="Harmonize a melody MIDI file")
    p_harm.add_argument("--model",  required=True, help="Trained model .json path")
    p_harm.add_argument("--input",  required=True, help="Input melody .mid path")
    p_harm.add_argument("--output", required=True, help="Output harmonized .mid path")
    p_harm.add_argument("--key",    default="C_major",
                        help="Key to harmonize in, e.g. C_major, D_minor (default: C_major)")

    # ── evaluate ─────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate", help="Evaluate model accuracy on POP909 test set")
    p_eval.add_argument("--model",       required=True, help="Trained model .json path")
    p_eval.add_argument("--pop909",      required=True, help="Path to POP909/POP909/ directory")
    p_eval.add_argument("--granularity", default="beat", choices=["beat", "measure"])
    p_eval.add_argument("--verbose",     action="store_true")

    args = parser.parse_args(argv)

    dispatch = {"train": cmd_train, "harmonize": cmd_harmonize, "evaluate": cmd_evaluate}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
