import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path
from types import MappingProxyType
from typing import Optional, Callable, Mapping, Collection
from urllib.parse import urljoin

from importlib_resources import read_text
from tqdm import tqdm

import aspset510.res
from aspset510.util import FSPath

# All parts of the dataset available for download.
ALL_PARTITION_FIELDS = MappingProxyType({
    'trainval': ('cameras', 'boxes', 'joints_3d', 'videos'),
    'test': ('cameras', 'boxes', 'videos'),
})
# Current version of the ASPset-510 dataset archives.
CURRENT_VERSION = 'v1'
# Buffer size for chunks of a file held in memory.
_BUFSIZE = shutil.COPY_BUFSIZE if hasattr(shutil, 'COPY_BUFSIZE') else 65536
# Names for archives that don't belong in a particular data partition.
COMMON_ARCHIVES = ['splits']


def read_checksums() -> Mapping[str, str]:
    """Read the expected MD5 checksum values for all ASPset-510 archive files.

    Returns:
        Mapping[str, str]: Mapping from filenames to hex-encoded MD5 checksums.
    """
    checksums = {}
    for line in read_text(aspset510.res, 'checksum.md5').splitlines(keepends=False):
        checksum, filename = line.split()
        checksums[filename] = checksum
    return checksums


def collect_archives(
    base_url: str,
    partition_fields: Mapping[str, Collection[str]] = ALL_PARTITION_FIELDS,
    version: str = CURRENT_VERSION,
) -> list:
    """Get information about each archive associated with `partition_fields`.

    Args:
        base_url (str): Base URL of a remote host providing the archives.
        partition_fields (Mapping[str, Collection[str]]): Specification of which archives to
            consider.
        version: Version of the archives to consider.

    Returns:
        A list of objects describing each archive.
    """
    checksums = read_checksums()
    if not base_url.endswith('/'):
        base_url += '/'
    archive_infos = []
    for partition, fields in partition_fields.items():
        for field in fields:
            filename = f'aspset510_{version}_{partition}-{field}.tar.gz'
            archive_infos.append({
                'filename': filename,
                'remote_url': urljoin(base_url, filename),
                'checksum': checksums[filename],
                'partition': partition,
                'field': field,
            })
    return archive_infos


def collect_common_archives(
    base_url: str,
    version: str = CURRENT_VERSION,
) -> list:
    """Get information about each archive not associated with a particular partition.

    Args:
        base_url (str): Base URL of a remote host providing the archives.
        version: Version of the archives to consider.

    Returns:
        A list of objects describing each archive.
    """
    checksums = read_checksums()
    if not base_url.endswith('/'):
        base_url += '/'
    archive_infos = []
    for archive_name in COMMON_ARCHIVES:
        filename = f'aspset510_{version}_common-{archive_name}.tar.gz'
        archive_infos.append({
            'filename': filename,
            'remote_url': urljoin(base_url, filename),
            'checksum': checksums[filename],
            'partition': '',
            'field': '',
        })
    return archive_infos


def download_file(
    dest_path: FSPath,
    src_url: str,
    progress: bool = False,
):
    """Download a file.

    Args:
        dest_path (FSPath): Local path to the desired file destination.
        src_url (str): URL for the remote file source.
        progress (bool): When set, the download progress will be shown.
    """
    dest_path = Path(dest_path)
    if progress:
        tqdm.write(f'Downloading {dest_path.name}...')
        bar = tqdm(leave=True, ascii=True, unit='B', unit_divisor=1024,
                   unit_scale=True)
        def update_progress(n, total):
            bar.total = total
            bar.update(n)
    else:
        bar = None
        update_progress = None
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


def calculate_md5_checksum(
    file_path: FSPath,
    callback: Optional[Callable[[int, int], None]] = None
) -> str:
    """Calculate the MD5 checksum for a file.

    Args:
        file_path (FSPath): Path to the file.
        callback (Optional[Callable[[int, int], None]]): Optional callback function for tracking
            calculation progress.

    Returns:
        The hex-encoded MD5 checksum.
    """
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
    file_path: FSPath,
    expected_checksum: str,
    progress: bool = False,
):
    """Verify the integrity of a file by calculating and comparing its MD5 checksum.

    Args:
        file_path (FSPath): Path to the file.
        expected_checksum (str): Expected hex-encoded MD5 checksum.
        progress (bool): When set, the calculation progress will be shown.

    Raises:
        ValueError: If the calculated MD5 checksum does not match `expected_checksum`.
    """
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
    actual = calculate_md5_checksum(file_path, update_progress)
    if actual != expected_checksum:
        raise ValueError(f'file integrity check failed for {file_name}')
    if bar is not None:
        bar.close()


def extract_tgz(
    archive_file: FSPath,
    data_dir: FSPath,
    strip_prefix: str = '',
    progress: bool = False,
):
    """Extract a gzipped tar archive file.

    Args:
        archive_file (FSPath): Path to the archive file to be extracted.
        data_dir (FSPath): Path to the destination directory for extracted files.
        strip_prefix (str): Common prefix to be stripped from archive member names.
        progress (bool): When set, the extraction progress will be shown.
    """
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
            if strip_prefix:
                member.name = str(Path(member.name).relative_to(strip_prefix))
            tar.extract(member, data_dir)
            if bar:
                bar.update(size)
        if bar:
            bar.close()


def extracted_files_exist(
    data_dir: FSPath,
    partition: str,
    field: str,
) -> bool:
    """Check whether extracted files exist for a particular field of ASPset-510.

    Args:
        data_dir (FSPath): ASPset-510 data directory path.
        partition (str): Partition to check (i.e. 'trainval' or 'test').
        field (str): Field to check (e.g. 'videos').

    Returns:
        `True` if the extracted files exist, `False` otherwise.
    """
    if not partition or not field:
        # We can only check fields that belong to a particular partition.
        return False
    data_dir = Path(data_dir)
    return data_dir.joinpath(partition, field).is_dir()


def download_and_extract_archives(
    data_dir: FSPath,
    archive_dir: FSPath,
    base_url: str,
    partition_fields: Mapping[str, Collection[str]] = ALL_PARTITION_FIELDS,
    version: str = CURRENT_VERSION,
    skip_existing: bool = True,
    skip_download_existing: bool = True,
    skip_checksum: bool = False,
    skip_extraction: bool = False,
    progress: bool = False,
):
    """Download and extract fields from ASPset-510.

    Args:
        data_dir (FSPath): ASPset-510 data directory path.
        archive_dir (FSPath): Directory for storing downloaded archives.
        base_url (str): Base URL of a remote host providing the archives.
        partition_fields (Mapping[str, Collection[str]]): Specification of which fields of the
            dataset to download and extract.
        version (str): Version of the dataset to consider.
        skip_existing (bool): When set, existing fields will be skipped.
        skip_download_existing (bool): When set, existing archives will be skipped.
        skip_checksum (bool): When set, checksums will not be verified.
        skip_extraction (bool): When set, archives will not be extracted.
        progress (bool): When set, the progress will be shown.
    """
    # Prepare local directories.
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(exist_ok=True, parents=True)
    # Get the archives to download and extract.
    archive_infos = collect_archives(base_url, partition_fields, version)
    archive_infos.extend(collect_common_archives(base_url, version))
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
            download_file(archive_file, url, progress=progress)
        if not skip_checksum:
            check_file_integrity(archive_file, checksum, progress=progress)
        if not skip_extraction:
            extract_tgz(archive_file, data_dir, strip_prefix='ASPset-510', progress=progress)
