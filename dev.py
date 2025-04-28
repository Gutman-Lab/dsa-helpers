"""Script to develop functions for the library."""

from argparse import ArgumentParser
from dsa_helpers.girder_utils import (
    login,
    post_annotation,
    remove_overlapping_annotations,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--api-token", type=str, default=None, help="Girder API token"
    )
    parser.add_argument(
        "--api-url", type=str, default=None, help="Girder API URL"
    )
    return parser.parse_args()


def main(args):
    gc = login(args.api_url, api_key=args.api_token)

    # Get an annotation document.
    doc = gc.get("annotation/...")

    annotation = remove_overlapping_annotations(
        doc["annotation"], ["Gray Matter", "White Matter", "Leptomeninges"]
    )

    # Push the new doc.
    annotation["name"] = "temp"

    response = post_annotation(gc, "...", annotation)

    print(response)


if __name__ == "__main__":
    main(parse_args())
