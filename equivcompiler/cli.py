"""Command-line interface for the representation compiler."""

from __future__ import annotations

import argparse
import json

from equivcompiler.checkpoint import convert_checkpoint


def _convert_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "convert-checkpoint",
        help="strictly migrate a checkpoint between exact execution backends",
    )
    parser.add_argument("--checkpoint", required=True, help="source checkpoint")
    parser.add_argument("--destination", required=True, help="converted checkpoint")
    parser.add_argument("--from", dest="from_backend", required=True)
    parser.add_argument("--to", dest="to_backend", required=True)
    parser.add_argument(
        "--output-representation",
        required=True,
        help="O(3) irreps or Cartesian symmetry formula",
    )
    parser.add_argument("--seed-irreps", required=True)
    parser.add_argument("--covariance", default="full")
    parser.add_argument(
        "--feature-scope", choices=("global", "node", "edge"), default="node"
    )
    parser.add_argument(
        "--output-scope",
        choices=("global", "node", "edge", "dense"),
        default="global",
        help="'dense' is a legacy alias for the declared feature scope",
    )
    parser.add_argument("--distribution", choices=("gaussian", "student_t"), default="gaussian")
    parser.add_argument("--budget", type=int, default=192)
    parser.add_argument("--low-rank", type=int, default=8)
    parser.add_argument("--audit-path")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(command="convert-checkpoint")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="equiv-compiler")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _convert_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "convert-checkpoint":
        audit = convert_checkpoint(
            args.checkpoint,
            args.destination,
            from_backend=args.from_backend,
            to_backend=args.to_backend,
            output=args.output_representation,
            seed_irreps=args.seed_irreps,
            covariance=args.covariance,
            feature_scope=args.feature_scope,
            output_scope=args.output_scope,
            distribution=args.distribution,
            budget=args.budget,
            low_rank=args.low_rank,
            audit_path=args.audit_path,
            overwrite=args.overwrite,
        )
        print(json.dumps(audit, indent=2))
        return 0
    raise RuntimeError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
