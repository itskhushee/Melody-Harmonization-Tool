"""Command-line interface — python -m harmonizer.cli <subcommand> ..."""

from __future__ import annotations
import argparse
import json
import sys


def cmd_train(args: argparse.Namespace) -> None:
    from harmonizer.train import estimate_parameters
    from harmonizer.harmonize import save_model

    with open(args.data) as f:
        corpus_raw = json.load(f)

    corpus = [(item["observations"], item["states"]) for item in corpus_raw]
    model = estimate_parameters(corpus, n_obs=12)
    save_model(model, args.out)
    print(f"Model saved to {args.out}")


def cmd_harmonize(args: argparse.Namespace) -> None:
    from harmonizer.harmonize import load_model, harmonize

    model = load_model(args.model)
    chords = harmonize(model, args.input)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(chords, f, indent=2)
        print(f"Chords written to {args.output}")
    else:
        print(" ".join(chords))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="harmonizer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Estimate HMM parameters from labeled corpus")
    p_train.add_argument("--data", required=True, help="Path to processed corpus JSON")
    p_train.add_argument("--out",  required=True, help="Output model .pkl path")

    p_harm = sub.add_parser("harmonize", help="Harmonize a melody MIDI file")
    p_harm.add_argument("--model",  required=True, help="Trained model .pkl path")
    p_harm.add_argument("--input",  required=True, help="Input melody .mid path")
    p_harm.add_argument("--output", default=None,  help="Output JSON path (optional)")

    args = parser.parse_args(argv)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "harmonize":
        cmd_harmonize(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
