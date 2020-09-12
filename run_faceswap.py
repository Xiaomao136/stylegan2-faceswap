#!/usr/bin/env Python
# coding=utf-8
import dnnlib.tflib as tflib
import argparse
import re
from faceswap.faceswap import FaceSwapper
import dnnlib


def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


_examples = '''examples:
    python faceswap.py --video=example.mp4 --style=style.png --col-styles="0-6"
'''


def main():
    parser = argparse.ArgumentParser(
        description='''faceswap''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser.add_argument('--video', help='video file', dest='video_file', required=True)
    parser.add_argument('--style', help='style file', dest='style_file', required=True)
    parser.add_argument('--col-styles', type=_parse_num_range,
                        help='Style layer range (default: %(default)s)', default='0-6')
    parser.add_argument('--start', type=int, help='start from', dest='start_from', default=0)
    args = parser.parse_args()

    network_pkl = args.network_pkl
    video_file = args.video_file
    style_file = args.style_file
    col_styles = args.col_styles
    start_from = args.start_from
    if network_pkl == 'None':
        network_pkl = None

    tflib.init_tf()
    swapper = FaceSwapper(network_pkl, video_file, style_file, col_styles)
    # swapper.face_swap(start_from)
    swapper.test2()


if __name__ == "__main__":
    main()
