#!/usr/bin/env python3
"""Run scDHF on one CSV dataset."""

from src.config import build_parser


def main():
    args = build_parser().parse_args()
    from src.pipeline import run

    run(args)


if __name__ == "__main__":
    main()
