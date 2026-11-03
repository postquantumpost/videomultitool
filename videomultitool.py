#!/usr/bin/env python3
"""videomultitool: a utility for operating on video files."""

import argparse
import os
import sys

# Hide TensorFlow's informational startup messages (used by RetinaFace/deepface).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from blankless_in_to_out import blankless_in_to_out
from filter_in_to_out import filter_in_to_out

vmt_debug = False
vmt_debug2 = False
slowok = False
do_blur = False
blankless = False
blur_edge = True
fill = False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Operate on video files."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug output",
    )
    parser.add_argument(
        "--debug2",
        action="store_true",
        help="enable second-level debug (random test blur boxes)",
    )
    parser.add_argument(
        "--slowok",
        action="store_true",
        help="allow slow processing functions (e.g. RetinaFace)",
    )
    parser.add_argument(
        "--nobluredge",
        dest="blur_edge",
        action="store_false",
        help="do not extend the blur with a fade beyond each box edge",
    )
    parser.add_argument(
        "--fill",
        action="store_true",
        help="fill each box with its average colour instead of blurring it",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--blur",
        dest="do_blur",
        action="store_true",
        help="blur detected faces, nudity and license plates (needs --input and --output)",
    )
    mode.add_argument(
        "--blankless",
        action="store_true",
        help="crop letterbox/pillarbox borders (needs --input and --output)",
    )
    parser.add_argument(
        "--input",
        metavar="FILE",
        help="input video file",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="output video file",
    )
    return parser.parse_args(argv)


def process_input(input_file, output_file=None):
    if not input_file:
        return
    print(f"Input file: {input_file}")
    if output_file:
        print(f"Output file: {output_file}")
        if blankless:
            blankless_in_to_out(input_file, output_file, debug=vmt_debug)
        elif do_blur:
            filter_in_to_out(input_file, output_file, debug=vmt_debug,
                             debug2=vmt_debug2, slowok=slowok,
                             blur_edge=blur_edge, fill=fill)


def main(argv=None):
    global vmt_debug, vmt_debug2, slowok, do_blur, blankless, blur_edge, fill
    args = parse_args(argv)
    vmt_debug = args.debug
    vmt_debug2 = args.debug2
    slowok = args.slowok
    do_blur = args.do_blur
    blankless = args.blankless
    blur_edge = args.blur_edge
    fill = args.fill

    if vmt_debug:
        print(f"Debug: arguments = {vars(args)}", file=sys.stderr)

    process_input(args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
