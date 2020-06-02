import numpy as np

from aspset510.evaluation import calculate_mpjpe, calculate_pck


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
