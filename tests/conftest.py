import os
from pathlib import Path

import numpy as np
import pytest

from aspset510 import Aspset510, Camera


@pytest.fixture
def aspset_data_path(request):
    default_path = Path(request.module.__file__).parent.parent.joinpath('data')
    path = Path(os.environ.get('ASPSET510_DATA_PATH', default_path)).absolute()
    if not path.is_dir():
        pytest.skip('cannot find Aspset data path (try setting the "ASPSET510_DATA_PATH" env var)')
    return path


@pytest.fixture
def aspset(aspset_data_path):
    return Aspset510(aspset_data_path)


@pytest.fixture
def clip(aspset):
    return aspset.clip('04ac', '0026')


@pytest.fixture
def camera():
    # This data is from 04ac-right.
    intrinsic_matrix = np.asarray([
        [3908.201416,    0.000000, 1907.136108, 0.000000],
        [   0.000000, 3904.395020, 1082.651855, 0.000000],
        [   0.000000,    0.000000,    1.000000, 0.000000],
    ])
    extrinsic_matrix = np.asarray([
        [ 0.355310, -0.037651, 0.933990, -16513.444863],
        [ 0.010982,  0.999288, 0.036106,   -694.394037],
        [-0.934684, -0.002572, 0.355470,  12560.579233],
        [ 0.000000,  0.000000, 0.000000,      1.000000],
    ])
    return Camera(intrinsic_matrix, extrinsic_matrix)


@pytest.fixture
def joints_3d():
    # This data is from 04ac-0026.
    return np.asarray([
        [-5.18131775e+02,  1.12147400e+03,  1.85388398e+04],
        [-4.73199890e+02,  7.00602722e+02,  1.85234902e+04],
        [-4.23036896e+02,  3.08107239e+02,  1.85275508e+04],
        [-2.80203583e+02,  1.95340317e+02,  1.84224238e+04],
        [-4.86079163e+02,  6.23998604e+01,  1.83367559e+04],
        [-3.91279541e+02, -1.83917984e+02,  1.83915391e+04],
        [-5.05070129e+02,  1.01266479e+03,  1.87362852e+04],
        [-2.35622971e+02,  6.76157654e+02,  1.86908340e+04],
        [-3.80700531e+02,  3.06988159e+02,  1.86746523e+04],
        [-2.01497665e+02,  9.60998917e+01,  1.86923438e+04],
        [-3.99593689e+02,  5.71454849e+01,  1.87760781e+04],
        [-4.34380341e+02, -1.88396454e+02,  1.87668672e+04],
        [-4.12053192e+02, -5.11558380e+02,  1.85718145e+04],
        [-3.89499725e+02, -3.58688873e+02,  1.85686406e+04],
        [-4.15598541e+02, -2.77202179e+02,  1.85831680e+04],
        [-4.56611267e+02,  6.73956985e+01,  1.86147578e+04],
        [-4.72049225e+02,  2.43091171e+02,  1.86294688e+04],
    ])


@pytest.fixture
def joints_2d():
    # This data is from 04ac-0026-right.
    return np.asarray([
        [2021.66769504, 1299.42125245],
        [2025.42582034, 1216.08135107],
        [2032.97349219, 1138.15800483],
        [2025.36732586, 1115.48445298],
        [1994.74109751, 1087.58712550],
        [2013.94834306, 1038.83602741],
        [2059.66599634, 1278.66144259],
        [2075.03514310, 1214.02638229],
        [2063.36670926, 1139.05195129],
        [2082.42262712, 1097.61661095],
        [2082.35171422, 1089.87025071],
        [2079.75317487, 1040.93403303],
        [2048.13982880, 974.88781273],
        [2048.16161323, 1005.33527749],
        [2048.19522635, 1021.75662551],
        [2048.23529122, 1090.62699530],
        [2048.42372027, 1125.62374581],
    ])
