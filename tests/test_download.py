from aspset510.download import read_checksums, calculate_md5_checksum


def test_read_checksums():
    checksums = read_checksums()
    assert isinstance(checksums, dict)
    assert len(checksums) > 0
    for filename, checksum in checksums.items():
        # A 128-bit MD5 hash is represented using 32 hexadecimal digits.
        assert len(checksum) == 32


def test_calculate_md5_checksum(tmp_path):
    expected = 'd8e8fca2dc0f896fd7cb4cb0031ba249'
    file_path = tmp_path.joinpath('test.txt')
    with file_path.open('w') as f:
        f.write('test\n')
    actual = calculate_md5_checksum(str(file_path))
    assert actual == expected
