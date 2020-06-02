import numpy as np
from numpy.testing import assert_allclose
from posekit.io import Mocap


def test_load_mocap(clip):
    mocap = clip.load_mocap()
    assert isinstance(mocap, Mocap)
    assert mocap.joint_positions.shape == (132, 17, 3)
    assert mocap.skeleton_name == 'aspset_17j'
    assert mocap.sample_rate == 50


def test_get_video_path(clip):
    expected = clip.aspset.data_dir.joinpath('trainval', 'videos', '04ac', '04ac-0026-left.mkv')
    assert clip.get_video_path('left') == expected


def test_load_camera_matrices(clip):
    intrinsic_matrix, extrinsic_matrix = clip.load_camera_matrices('left')
    assert intrinsic_matrix.shape == (3, 4)
    assert_allclose(extrinsic_matrix, np.eye(4))


def test_load_camera(clip):
    camera = clip.load_camera('left')
    assert camera.intrinsic_matrix.shape == (3, 4)
    assert_allclose(camera.extrinsic_matrix, np.eye(4))
