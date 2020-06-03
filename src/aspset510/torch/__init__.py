from dataclasses import dataclass
from typing import Collection

import numpy as np
from glupy.math import ensure_cartesian
from torch.utils.data import Dataset
from tvl import VideoLoader

from aspset510 import Aspset510, Clip, Camera
from aspset510.geometry import zoom_roi, square_containing_rectangle
from aspset510.util import FSPath

ALL_FIELDS = ('cameras', 'boxes', 'joints_3d', 'joints_2d', 'videos')


@dataclass
class _ExampleRef:
    clip: Clip
    frame_index: int
    camera_id: str


class Aspset510Dataset(Dataset):
    def __init__(self, clips: Collection[Clip], fields: Collection[str] = ALL_FIELDS):
        """Create a dataset instance for loading examples from a collection of ASPset-510 clips.

        Args:
            clips (Collection[Clip]): Clips to load examples from.
            fields (Collection[str]): List of dataset fields to derive example annotations from.
                For example, 'cameras' will include camera parameters, 'videos' will include the
                video frame image, etc. If image data is not needed, it is highly recommended to
                omit 'videos' for greatly improved data loading performance.
        """
        refs = []
        for clip in clips:
            for i in range(clip.frame_count):
                for camera_id in Aspset510.CAMERA_IDS:
                    refs.append(_ExampleRef(clip, i, camera_id))
        self._refs = refs
        self.fields = fields

    def __len__(self):
        return len(self._refs)

    def _fill_entries_from_cameras(self, example, ref):
        if 'intrinsic_matrix' in example and 'extrinsic_matrix' in example:
            return
        intrinsic_matrix, extrinsic_matrix = ref.clip.load_camera_matrices(ref.camera_id)
        example['intrinsic_matrix'] = intrinsic_matrix
        example['extrinsic_matrix'] = extrinsic_matrix

    def _fill_entries_from_joints_3d(self, example, ref):
        if 'joints_3d' in example and 'skeleton_name' in example:
            return
        self._fill_entries_from_cameras(example, ref)
        mocap = ref.clip.load_mocap()
        camera = Camera(example['intrinsic_matrix'], example['extrinsic_matrix'])
        example['joints_3d'] = ensure_cartesian(camera.world_to_camera_space(mocap.joint_positions[ref.frame_index]), d=3)
        example['skeleton_name'] = mocap.skeleton_name

    def _fill_entries_from_joints_2d(self, example, ref):
        if 'joints_2d' in example:
            return
        self._fill_entries_from_cameras(example, ref)
        self._fill_entries_from_joints_3d(example, ref)
        camera = Camera(example['intrinsic_matrix'], example['extrinsic_matrix'])
        example['joints_2d'] = ensure_cartesian(camera.camera_to_image_space(example['joints_3d']), d=2)

    def _fill_entries_from_boxes(self, example, ref):
        if 'box' in example and 'crop_box' in example:
            return
        box = ref.clip.load_bounding_boxes(ref.camera_id)[ref.frame_index]
        crop_box = np.asarray(square_containing_rectangle(zoom_roi(box, zoom=2/3)))
        example['box'] = box
        example['crop_box'] = crop_box

    def load_video_frame_image(self, video_path, frame_index):
        vl = VideoLoader(video_path, 'cpu')
        return vl.select_frame(frame_index)

    def _fill_entries_from_videos(self, example, ref):
        if 'image' in example:
            return
        video_path = ref.clip.get_video_path(ref.camera_id)
        example['image'] = self.load_video_frame_image(video_path, ref.frame_index)

    def __getitem__(self, index):
        ref = self._refs[index]
        example = {
            'index': index,
            'subject_id': ref.clip.subject_id,
            'clip_id': ref.clip.clip_id,
            'frame_index': ref.frame_index,
            'camera_id': ref.camera_id,
        }
        for field in self.fields:
            if field == 'cameras':
                self._fill_entries_from_cameras(example, ref)
            if field == 'joints_3d':
                self._fill_entries_from_joints_3d(example, ref)
            if field == 'joints_2d':
                self._fill_entries_from_joints_2d(example, ref)
            if field == 'boxes':
                self._fill_entries_from_boxes(example, ref)
            if field == 'videos':
                self._fill_entries_from_videos(example, ref)
        return example


def create_aspset510_dataset(data_dir: FSPath, split: str, **kwargs) -> Aspset510Dataset:
    """Create a PyTorch Dataset instance for a particular ASPset-510 split.

    Args:
        data_dir (FSPath): ASPset-510 data directory path.
        split (str): Split of the dataset (e.g. 'train' or 'test').
        **kwargs: Keyword arguments to pass to the `Aspset510Dataset` constructor.

    Returns:
        Aspset510Dataset: The dataset instance.
    """
    aspset = Aspset510(data_dir)
    clips = aspset.clips(split)
    return Aspset510Dataset(clips, **kwargs)
