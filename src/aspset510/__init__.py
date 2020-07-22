import csv
import json
from pathlib import Path

import numpy as np
from posekit.io import load_mocap, Mocap

from aspset510.camera import Camera


class Aspset510:
    ALL_CAMERA_IDS = ['left', 'mid', 'right']

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self._read_splits()

    def _read_splits(self):
        """Read the CSV file describing which split (train, val, or test) each clip belongs to.
        """
        self.splits = {}
        self._inv_splits = {}
        self.cameras_for_clip = {}
        with self.data_dir.joinpath('splits.csv').open('r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                subject_id, clip_id, split, camera_id = row
                if camera_id == 'all':
                    cameras = self.ALL_CAMERA_IDS
                else:
                    cameras = [camera_id]
                self.cameras_for_clip[(subject_id, clip_id)] = cameras
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

    def clips(self, split):
        if split == 'train':
            return self.train_clips()
        if split == 'val':
            return self.val_clips()
        if split == 'trainval':
            return self.trainval_clips()
        if split == 'test':
            return self.test_clips()
        if split == 'all':
            return self.all_clips()
        raise ValueError(split)


class Clip:
    def __init__(self, aspset: Aspset510, subject_id: str, clip_id: str):
        self.aspset = aspset
        self._subject_id = subject_id
        self._clip_id = clip_id

    def __repr__(self):
        return f'Clip(subject_id={self.subject_id}, clip_id={self.clip_id})'

    def __eq__(self, other):
        if isinstance(other, Clip):
            return self.subject_id == other.subject_id and self.clip_id == other.clip_id
        return False

    def __hash__(self):
        return hash((self.subject_id, self.clip_id))

    @property
    def subject_id(self):
        return self._subject_id

    @property
    def clip_id(self):
        return self._clip_id

    @property
    def camera_ids(self):
        return self.aspset.cameras_for_clip[(self.subject_id, self.clip_id)]

    @property
    def split(self):
        """Get the split (i.e. "train", "val", or "test") that this clip belongs to.
        """
        return self.aspset.find_split(self.subject_id, self.clip_id)

    def _rel_path(self, *args):
        return self.aspset.data_dir.joinpath('test' if self.split == 'test' else 'trainval', *args)

    def load_mocap(self) -> Mocap:
        c3d_file = self._rel_path('joints_3d', self.subject_id, f'{self.subject_id}-{self.clip_id}.c3d')
        if not c3d_file.is_file():
            raise FileNotFoundError(str(c3d_file))
        return load_mocap(c3d_file)

    def get_video_path(self, camera_id) -> Path:
        assert camera_id in self.camera_ids
        basename = f'{self.subject_id}-{self.clip_id}-{camera_id}.mkv'
        video_path = self._rel_path('videos', self.subject_id, basename)
        return video_path

    def load_camera_matrices(self, camera_id):
        assert camera_id in self.camera_ids
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

    def load_bounding_boxes(self, camera_id):
        basename = f'{self.subject_id}-{self.clip_id}-{camera_id}.csv'
        boxes_path = self._rel_path('boxes', self.subject_id, basename)
        boxes = []
        with boxes_path.open('r', newline='') as f:
            reader = csv.reader(f)
            # Discard header row.
            next(reader)
            # Read bounding boxes.
            for row in reader:
                boxes.append([float(e) for e in row])
        return np.asarray(boxes, dtype=np.float32)

    @property
    def frame_count(self):
        return len(self.load_bounding_boxes(self.camera_ids[0]))
