from pathlib import Path

import numpy as np
import pytest
from posekit.io import Mocap, save_mocap

from aspset510.evaluation import calculate_mpjpe, calculate_pck, find_and_load_prediction


def test_calculate_mpjpe():
    actual_poses = np.asarray([
        [[1, 1, 1], [0, 0, 2]]
    ])
    expected_poses = np.asarray([
        [[1, 1, 1], [3, 4, 2]]
    ])
    mpjpe = calculate_mpjpe(actual_poses, expected_poses)
    assert mpjpe == 2.5


def test_calculate_pck():
    actual_poses = np.asarray([
        [[1, 1, 1], [0, 0, 2]]
    ])
    expected_poses = np.asarray([
        [[1, 1, 1], [3, 4, 2]]
    ])
    pck = calculate_pck(actual_poses, expected_poses, threshold=4)
    assert pck == 0.5


class TestFindAndLoadPrediction:
    @pytest.fixture
    def preds_dir(self, tmpdir):
        mocap = Mocap(np.zeros((5, 17, 3)), 'aspset_17j', 50)
        filenames = [
            '1e28-0001.c3d',
            '1e28-0001-right.c3d',
            '8a59-0035-left.c3d',
        ]
        p = Path(tmpdir.mkdir('preds'))
        for filename in filenames:
            save_mocap(mocap, p.joinpath(filename))
        return p

    def test_found(self, preds_dir):
        find_and_load_prediction(preds_dir, '8a59', '0035', 'left', False)

    def test_found_ignoring_unknown_camera(self, preds_dir):
        find_and_load_prediction(preds_dir, '1e28', '0001', 'right', False)

    def test_found_unknown_camera(self, preds_dir):
        find_and_load_prediction(preds_dir, '1e28', '0001', 'left', True)

    def test_not_found(self, preds_dir):
        with pytest.raises(RuntimeError) as ex:
            find_and_load_prediction(preds_dir, '8a59', '0035', 'right', False)
        assert str(ex.value) == 'no prediction file found for 8a59-0035-right'

    def test_found_multiple_unknown_camera(self, preds_dir):
        with pytest.raises(RuntimeError) as ex:
            find_and_load_prediction(preds_dir, '1e28', '0001', 'right', True)
        assert str(ex.value) == 'multiple prediction files found for 1e28-0001-right'
