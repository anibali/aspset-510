from dataclasses import dataclass
from typing import Collection, Optional, Tuple

import numpy as np
import torch
from aspset510 import Aspset510, Clip, Camera, constants
from aspset510.geometry import zoom_roi, square_containing_rectangle
from aspset510.util import FSPath
from glupy.math import ensure_cartesian
from torch.utils.data import Dataset
from tvl import VideoLoader

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
            clips: Clips to load examples from.
            fields: List of dataset fields to derive example annotations from.
                For example, 'cameras' will include camera parameters, 'videos' will include the
                video frame image, etc. If image data is not needed, it is highly recommended to
                omit 'videos' for greatly improved data loading performance.
        """
        refs = []
        for clip in clips:
            for camera_id in clip.camera_ids:
                for i in range(clip.frame_count):
                    refs.append(_ExampleRef(clip, i, camera_id))
        self._refs = refs
        self.fields = fields
        self.fps = constants.FPS

    def __len__(self):
        return len(self._refs)

    def _get_ref(self, index, dt) -> _ExampleRef:
        unmodified_ref = self._refs[index]
        frame_index = int(round(unmodified_ref.frame_index + self.fps * dt))
        frame_index = min(max(frame_index, 0), unmodified_ref.clip.frame_count - 1)
        ref = _ExampleRef(unmodified_ref.clip, frame_index, unmodified_ref.camera_id)
        return ref

    def get_unique_video_id(self, index: int, dt: float) -> str:
        """Get a unique identifier for the video corresponding to a particular example.

        Args:
            index: The index of the example.
            dt: A time offset relative to the original example location.

        Returns:
            The unique video ID string.
        """
        ref = self._get_ref(index, dt)
        return f'{ref.clip.subject_id}-{ref.clip.clip_id}-{ref.camera_id}'

    def get_camera_id(self, index, dt) -> str:
        """Get the corresponding camera ID for a particular example.

        Args:
            index: The index of the example.
            dt: A time offset relative to the original example location.

        Returns:
            The camera ID.
        """
        return self._get_ref(index, dt).camera_id

    def get_frame_index(self, index, dt) -> int:
        """Get the corresponding video frame index for a particular example.

        Args:
            index: The index of the example.
            dt: A time offset relative to the original example location.

        Returns:
            The video frame index.
        """
        ref = self._get_ref(index, dt)
        return ref.frame_index

    def get_clip(self, index, dt) -> Clip:
        """Get the corresponding clip for a particular example.

        Args:
            index: The index of the example.
            dt: A time offset relative to the original example location.

        Returns:
            The clip.
        """
        return self._get_ref(index, dt).clip

    def load_camera_matrices(self, index: int, dt: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Get the camera intrinsic and extrinsic matrices for a particular example.

        Args:
            index: The index of the example.
            dt: A time offset relative to the original example location.

        Returns:
            A tuple containing the camera intrinsic (3x4) and extrinsic (4x4) matrices.
        """
        ref = self._get_ref(index, dt)
        return ref.clip.load_camera_matrices(ref.camera_id)

    def _fill_entries_from_cameras(self, example, index, dt):
        if 'intrinsic_matrix' in example and 'extrinsic_matrix' in example:
            return
        intrinsic_matrix, extrinsic_matrix = self.load_camera_matrices(index, dt)
        example['intrinsic_matrix'] = intrinsic_matrix
        example['extrinsic_matrix'] = extrinsic_matrix

    def load_joints_3d(
        self,
        index: int,
        dt: float,
        *,
        camera: Optional[Camera] = None,
    ) -> Tuple[np.ndarray, str]:
        """Get the 3D joint locations for a particular example.

        The joints will be in camera space (not world space) coordinates.

        Args:
            index: The index of the example.
            dt: A time offset relative to the original example location.
            camera: The camera for this example (will be automatically loaded if not specified).

        Returns:
            A tuple containing the 3D joint locations in camera space and the skeleton description
            name.
        """
        ref = self._get_ref(index, dt)
        if camera is None:
            camera = Camera(*self.load_camera_matrices(index, dt))
        mocap = ref.clip.load_mocap()
        joints_3d = ensure_cartesian(camera.world_to_camera_space(mocap.joint_positions[ref.frame_index]), d=3)
        return joints_3d, mocap.skeleton_name

    def _fill_entries_from_joints_3d(self, example, index, dt):
        if 'joints_3d' in example and 'skeleton_name' in example:
            return
        self._fill_entries_from_cameras(example, index, dt)
        camera = Camera(example['intrinsic_matrix'], example['extrinsic_matrix'])
        joints_3d, skeleton_name = self.load_joints_3d(index, dt, camera=camera)
        example['joints_3d'] = joints_3d
        example['skeleton_name'] = skeleton_name

    def load_joints_2d(
        self,
        index: int,
        dt: float,
        *,
        camera: Camera = None,
        joints_3d: np.ndarray = None,
        skeleton_name: str = None,
    ) -> Tuple[np.ndarray, str]:
        """Get the 2D joint locations for a particular example.

        The joints will be in image space coordinates.

        Args:
            index: The index of the example.
            dt: A time offset relative to the original example location.
            camera: The camera for this example (will be automatically loaded if not specified).
            joints_3d: The camera space 3D joint locations for this example (will be automatically
                loaded if not specified).
            skeleton_name: The skeleton name for this example (will be automatically
                loaded if not specified).

        Returns:
            A tuple containing the 2D joint locations in image space and the skeleton description
            name.
        """
        if camera is None:
            camera = Camera(*self.load_camera_matrices(index, dt))
        if joints_3d is None or skeleton_name is None:
            joints_3d, skeleton_name = self.load_joints_3d(index, dt, camera=camera)
        return ensure_cartesian(camera.camera_to_image_space(joints_3d), d=2), skeleton_name

    def _fill_entries_from_joints_2d(self, example, index, dt):
        if 'joints_2d' in example:
            return
        self._fill_entries_from_cameras(example, index, dt)
        self._fill_entries_from_joints_3d(example, index, dt)
        joints_2d, _ = self.load_joints_2d(
            index,
            dt,
            camera=Camera(example['intrinsic_matrix'], example['extrinsic_matrix']),
            joints_3d=example['joints_3d'],
            skeleton_name=example['skeleton_name'],
        )
        example['joints_2d'] = joints_2d

    def _fill_entries_from_boxes(self, example, index, dt):
        if 'box' in example and 'crop_box' in example:
            return
        ref = self._get_ref(index, dt)
        box = ref.clip.load_bounding_boxes(ref.camera_id)[ref.frame_index]
        crop_box = np.asarray(square_containing_rectangle(zoom_roi(box, zoom=2/3)))
        example['box'] = box
        example['crop_box'] = crop_box

    def load_image(self, index: int, dt: float) -> torch.Tensor:
        """Get the video frame image for a particular example.

        Args:
            index: The index of the example.
            dt: A time offset relative to the original example location.

        Returns:
            The video frame image.
        """
        ref = self._get_ref(index, dt)
        video_path = ref.clip.get_video_path(ref.camera_id)
        vl = VideoLoader(video_path, 'cpu')
        return vl.select_frame(ref.frame_index)

    def _fill_entries_from_videos(self, example, index, dt):
        if 'image' in example:
            return
        example['image'] = self.load_image(index, dt)

    def __getitem__(self, index):
        dt = 0.0
        ref = self._get_ref(index, dt)
        example = {
            'index': index,
            'subject_id': ref.clip.subject_id,
            'clip_id': ref.clip.clip_id,
            'frame_index': ref.frame_index,
            'camera_id': ref.camera_id,
        }
        for field in self.fields:
            if field == 'cameras':
                self._fill_entries_from_cameras(example, index, dt)
            if field == 'joints_3d':
                self._fill_entries_from_joints_3d(example, index, dt)
            if field == 'joints_2d':
                self._fill_entries_from_joints_2d(example, index, dt)
            if field == 'boxes':
                self._fill_entries_from_boxes(example, index, dt)
            if field == 'videos':
                self._fill_entries_from_videos(example, index, dt)
        return example


def create_aspset510_dataset(data_dir: FSPath, split: str, **kwargs) -> Aspset510Dataset:
    """Create a PyTorch Dataset instance for a particular ASPset-510 split.

    Args:
        data_dir: ASPset-510 data directory path.
        split: Split of the dataset (e.g. 'train' or 'test').
        **kwargs: Keyword arguments to pass to the `Aspset510Dataset` constructor.

    Returns:
        The dataset instance.
    """
    aspset = Aspset510(data_dir)
    clips = aspset.clips(split)
    return Aspset510Dataset(clips, **kwargs)
