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

ALL_PARTITIONS = ('trainval', 'test')
ALL_FIELDS = ('cameras', 'joints_3d', 'videos')
CURRENT_VERSION = 'v1'
_BUFSIZE = shutil.COPY_BUFSIZE if hasattr(shutil, 'COPY_BUFSIZE') else 65536


def read_checksums():
    checksums = {}
    for line in read_text(aspset510.res, 'checksum.md5').splitlines(keepends=False):
        checksum, filename = line.split()
        checksums[filename] = checksum
    return checksums


def collect_archives(
    base_url: str,
    partitions: Collection[str] = ALL_PARTITIONS,
    fields: Collection[str] = ALL_FIELDS,
    version: str = CURRENT_VERSION,
):
    checksums = read_checksums()
    if not base_url.endswith('/'):
        base_url += '/'
    archive_infos = []
    for partition, field in product(partitions, fields):
        filename = f'aspset510_{version}_{partition}-{field}.tar.gz'
        archive_infos.append({
            'filename': filename,
            'remote_url': urljoin(base_url, filename),
            'checksum': checksums[filename],
            'partition': partition,
            'field': field,
        })
    return archive_infos


def download_file(
    dest_path,
    src_url: str,
    filename: str,
    progress: bool = False,
):
    if progress:
        tqdm.write(f'Downloading {filename}...')
        bar = tqdm(leave=True, ascii=True, unit='B', unit_divisor=1024,
                   unit_scale=True)
        def update_progress(n, total):
            bar.total = total
            bar.update(n)
    else:
        bar = None
        update_progress = None
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
            if update_progress:
                update_progress(len(buf), fsrc_size)
    if bar is not None:
        bar.close()


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


def check_file_integrity(
    file_path,
    expected_checksum: str,
    progress: bool = False,
):
    file_path = Path(file_path)
    file_name = file_path.name
    if progress:
        tqdm.write(f'Checking {file_name}...')
        bar = tqdm(leave=True, ascii=True, unit='B', unit_divisor=1024,
                   unit_scale=True)
        def update_progress(n, total):
            bar.total = total
            bar.update(n)
    else:
        bar = None
        update_progress = None
    actual = compute_md5_checksum(file_path, update_progress)
    if actual != expected_checksum:
        raise ValueError(f'file integrity check failed for {file_name}')
    if bar is not None:
        bar.close()


def extract_tgz(
    archive_file,
    data_dir,
    gobble_prefix: str = '',
    progress: bool = False,
):
    archive_file = Path(archive_file)
    data_dir = Path(data_dir)
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
            if gobble_prefix:
                member.name = str(Path(member.name).relative_to(gobble_prefix))
            tar.extract(member, data_dir)
            if bar:
                bar.update(size)
        if bar:
            bar.close()


def extracted_files_exist(
    data_dir,
    partition: str,
    field: str,
):
    data_dir = Path(data_dir)
    return data_dir.joinpath(partition, field).is_dir()


def download_and_extract_archives(
    data_dir,
    archive_dir,
    base_url: str,
    partitions: Collection[str] = ALL_PARTITIONS,
    fields: Collection[str] = ALL_FIELDS,
    version: str = CURRENT_VERSION,
    skip_existing: bool = True,
    skip_download_existing: bool = True,
    skip_checksum: bool = False,
    skip_extraction: bool = False,
    progress: bool = False,
):
    # Prepare local directories.
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(exist_ok=True, parents=True)
    # Get the archives to download and extract.
    archive_infos = collect_archives(base_url, partitions, fields, version)
    for archive_info in archive_infos:
        filename = archive_info['filename']
        url = archive_info['remote_url']
        checksum = archive_info['checksum']
        archive_file = archive_dir.joinpath(filename)
        if skip_existing and extracted_files_exist(data_dir, archive_info['partition'], archive_info['field']):
            if progress:
                tqdm.write(f'Skipping {filename}...')
            continue
        if not skip_download_existing or not archive_file.exists():
            download_file(archive_file, url, filename, progress=progress)
        if not skip_checksum:
            check_file_integrity(archive_file, checksum, progress=progress)
        if not skip_extraction:
            extract_tgz(archive_file, data_dir, gobble_prefix='ASPset-510', progress=progress)
