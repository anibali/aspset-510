"""Download and extract the ASPset-510 dataset.
"""

import argparse
import sys
from pathlib import Path

from aspset510.download import download_and_extract_archives
from aspset510.util import add_boolean_argument


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True,
                        help='path to the base dataset directory')
    parser.add_argument('--archive-dir', type=str, default=None,
                        help='path to the downloaded dataset archives (default: {DATA_DIR}/archives)')
    parser.add_argument('--mirror', type=str, required=True,
                        help='base URL of the archive mirror')
    add_boolean_argument(parser, 'skip_existing', default=True,
                         description='skip downloading and extracting archives which have been extracted previously')
    add_boolean_argument(parser, 'skip_download_existing', default=True,
                         description='skip downloading existing archives')
    add_boolean_argument(parser, 'skip_checksum', default=False,
                         description='skip checking archive integrity')
    add_boolean_argument(parser, 'skip_extraction', default=False,
                         description='skip extracting files')
    return parser


def main(args):
    opts = argument_parser().parse_args(args)

    data_dir = Path(opts.data_dir)
    if opts.archive_dir:
        archive_dir = Path(opts.archive_dir)
    else:
        archive_dir = data_dir.joinpath('archives')

    download_and_extract_archives(
        data_dir=data_dir,
        archive_dir=archive_dir,
        base_url=opts.mirror,
        fields=['cameras'],
        skip_existing=opts.skip_existing,
        skip_download_existing=opts.skip_download_existing,
        skip_checksum=opts.skip_checksum,
        skip_extraction=opts.skip_extraction,
        progress=True,
    )


if __name__ == '__main__':
    main(sys.argv[1:])
