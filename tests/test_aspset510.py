from pathlib import Path


def test_smoke(aspset):
    assert isinstance(aspset.data_dir, Path)


def test_all_clips(aspset):
    # Ensure that the ASPset-510 dataset lives up to its name ;)
    assert len(aspset.all_clips()) == 510
