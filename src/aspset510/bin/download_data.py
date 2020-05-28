"""Download and extract the ASPset-510 dataset.
"""

import argparse
import sys
from pathlib import Path

from aspset510.download import download_archives, check_archives, extract_archives
from aspset510.util import add_boolean_argument


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True,
                        help='path to the base dataset directory')
    parser.add_argument('--archive-dir', type=str, default=None,
                        help='path to the downloaded dataset archives (default: {DATA_DIR}/archives)')
    parser.add_argument('--mirror', type=str, required=True,
                        help='base URL of the archive mirror')
    add_boolean_argument(parser, 'skip_download_existing', default=True,
                         description='skip downloading existing files')
    add_boolean_argument(parser, 'skip_checksum', default=False,
                         description='skip checking file integrity')
    return parser


def main(args):
    opts = argument_parser().parse_args(args)

    data_dir = Path(opts.data_dir)
    if opts.archive_dir:
        archive_dir = Path(opts.archive_dir)
    else:
        archive_dir = data_dir.joinpath('archives')

    download_archives(archive_dir, opts.mirror, parts=['cameras'],
                      skip_existing=opts.skip_download_existing, progress=True)
    if not opts.skip_checksum:
        check_archives(archive_dir, progress=True)
    extract_archives(data_dir, archive_dir, progress=True)


if __name__ == '__main__':
    main(sys.argv[1:])
