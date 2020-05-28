import hashlib
import shutil
import tarfile
import urllib.request
from itertools import product
from pathlib import Path
from typing import Collection, Optional, Callable
from urllib.parse import urljoin

from importlib_resources import read_text
from tqdm import tqdm

import aspset510.res

_BUFSIZE = shutil.COPY_BUFSIZE if hasattr(shutil, 'COPY_BUFSIZE') else 65536


def download_archive(src_url, dest_path, callback: Optional[Callable[[int, int], None]] = None):
    dest_path = Path(dest_path)
    with urllib.request.urlopen(src_url) as src_file, dest_path.open('wb') as dest_file:
        fsrc_read = src_file.read
        fdst_write = dest_file.write
        fsrc_size = int(src_file.info()['Content-Length'] or 0)

        while True:
            buf = fsrc_read(_BUFSIZE)
            if not buf:
                break
            fdst_write(buf)
            if callback:
                callback(len(buf), fsrc_size)


def download_archives(
    archive_dir,
    base_url: str,
    subsets: Collection[str] = ('trainval', 'test'),
    parts: Collection[str] = ('cameras', 'joints_3d', 'videos'),
    version: str = 'v1',
    skip_existing: bool = True,
    progress: bool = False,
):
    if not base_url.endswith('/'):
        base_url += '/'
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(exist_ok=True, parents=True)

    for subset, part in product(subsets, parts):
        basename = f'aspset510_{version}_{subset}-{part}.tar.gz'
        dest_path = archive_dir.joinpath(basename)
        if skip_existing and dest_path.exists():
            continue
        src_url = urljoin(base_url, basename)
        if progress:
            tqdm.write(f'Downloading {basename}...')
            bar = tqdm(leave=True, ascii=True, unit='B', unit_divisor=1024,
                       unit_scale=True)
            def update_progress(n, total):
                bar.total = total
                bar.update(n)
        else:
            bar = None
            update_progress = None
        download_archive(src_url, dest_path, update_progress)
        if bar is not None:
            bar.close()


def _list_archives(archive_dir: Path):
    return archive_dir.glob('aspset510_*.tar.gz')


def read_checksums():
    checksums = {}
    for line in read_text(aspset510.res, 'checksum.md5').splitlines(keepends=False):
        checksum, filename = line.split()
        checksums[filename] = checksum
    return checksums


def compute_md5_checksum(file_path, callback: Optional[Callable[[int, int], None]] = None):
    file_path = Path(file_path)
    file_size = file_path.stat().st_size
    md5 = hashlib.md5()
    with file_path.open('rb') as f:
        for chunk in iter(lambda: f.read(_BUFSIZE), b''):
            md5.update(chunk)
            if callback:
                callback(len(chunk), file_size)
    return md5.hexdigest()


def check_archives(
    archive_dir,
    progress: bool = False,
):
    archive_dir = Path(archive_dir)
    checksums = read_checksums()
    for archive_file in _list_archives(archive_dir):
        archive_name = archive_file.name
        if progress:
            tqdm.write(f'Checking {archive_name}...')
            bar = tqdm(leave=True, ascii=True, unit='B', unit_divisor=1024,
                       unit_scale=True)
            def update_progress(n, total):
                bar.total = total
                bar.update(n)
        else:
            bar = None
            update_progress = None
        expected = checksums[archive_name]
        actual = compute_md5_checksum(archive_file, update_progress)
        if actual != expected:
            raise ValueError(f'file integrity check failed for {archive_name}')
        if bar is not None:
            bar.close()


def extract_archives(
    data_dir,
    archive_dir,
    progress: bool = False,
):
    data_dir = Path(data_dir)
    archive_dir = Path(archive_dir)
    for archive_file in _list_archives(archive_dir):
        with tarfile.open(archive_file, 'r:gz') as tar:
            members = [m for m in tar.getmembers() if m.isreg()]
            sizes = [m.size for m in members]
            total_size = sum(sizes)
            if progress:
                tqdm.write(f'Extracting {archive_file.name}...')
                bar = tqdm(total=total_size, leave=True, ascii=True, unit='B', unit_divisor=1024,
                           unit_scale=True)
            else:
                bar = None
            for member, size in zip(members, sizes):
                member.name = str(Path(member.name).relative_to('ASPset-510'))
                tar.extract(member, data_dir)
                if bar:
                    bar.update(size)
            if bar:
                bar.close()
