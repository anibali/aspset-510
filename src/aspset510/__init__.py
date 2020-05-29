import csv
import json
from pathlib import Path

import numpy as np
from posekit.io import load_c3d_mocap, Mocap

from aspset510.camera import Camera


class Aspset510:
    CAMERA_IDS = ['left', 'mid', 'right']

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self._read_splits()

    def _read_splits(self):
        """Read the CSV file describing which split (train, val, or test) each clip belongs to.
        """
        self.splits = {}
        self._inv_splits = {}
        with self.data_dir.joinpath('splits.csv').open('r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                subject_id, clip_id, split = row
                self.splits.setdefault(split, []).append([subject_id, clip_id])
                self._inv_splits[(subject_id, clip_id)] = split

    @property
    def split_names(self):
        return list(self.splits.keys())

    def find_split(self, subject_id: str, clip_id: str) -> str:
        return self._inv_splits[(subject_id, clip_id)]

    def clip(self, subject_id: str, clip_id: str) -> 'Clip':
        return Clip(self, subject_id, clip_id)

    def _split_clips(self, split):
        return [
            self.clip(subject_id, clip_id)
            for subject_id, clip_id in self.splits[split]
        ]

    def train_clips(self):
        return self._split_clips('train')

    def val_clips(self):
        return self._split_clips('val')

    def trainval_clips(self):
        return self.train_clips() + self.val_clips()

    def test_clips(self):
        return self._split_clips('test')

    def all_clips(self):
        return self.trainval_clips() + self.test_clips()


class Clip:
    def __init__(self, aspset: Aspset510, subject_id: str, clip_id: str):
        self.aspset = aspset
        self.subject_id = subject_id
        self.clip_id = clip_id

    def __repr__(self):
        return f'Clip(subject_id={self.subject_id}, clip_id={self.clip_id})'

    @property
    def split(self):
        return self.aspset.find_split(self.subject_id, self.clip_id)

    def _rel_path(self, *args):
        return self.aspset.data_dir.joinpath('test' if self.split == 'test' else 'trainval', *args)

    def load_mocap(self) -> Mocap:
        c3d_file = self._rel_path('joints_3d', self.subject_id, f'{self.subject_id}-{self.clip_id}.c3d')
        return load_c3d_mocap(c3d_file)

    def get_video_path(self, camera_id) -> Path:
        assert camera_id in self.aspset.CAMERA_IDS
        basename = f'{self.subject_id}-{self.clip_id}-{camera_id}.mkv'
        video_path = self._rel_path('videos', self.subject_id, basename)
        return video_path

    def load_camera_matrices(self, camera_id):
        assert camera_id in self.aspset.CAMERA_IDS
        basename = f'{self.subject_id}-{camera_id}.json'
        camera_path = self._rel_path('cameras', self.subject_id, basename)
        with camera_path.open('r') as f:
            camera_json = json.load(f)
        intrinsic_matrix = np.asarray(camera_json['intrinsic_matrix']).reshape([3, 4])
        extrinsic_matrix = np.asarray(camera_json['extrinsic_matrix']).reshape([4, 4])
        return intrinsic_matrix, extrinsic_matrix

    def load_camera(self, camera_id):
        intrinsic_matrix, extrinsic_matrix = self.load_camera_matrices(camera_id)
        return Camera(intrinsic_matrix, extrinsic_matrix)
