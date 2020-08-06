import pytest

from aspset510.torch import create_aspset510_dataset


class TestAspset510Dataset():
    @pytest.fixture
    def val_dataset(self, aspset_data_path):
        return create_aspset510_dataset(aspset_data_path, 'val')

    def test_len(self, val_dataset):
        assert len(val_dataset) == 45288

    def test_getitem(self, val_dataset):
        example = val_dataset[0]
        assert example['index'] == 0
        assert example['subject_id'] == '4d9e'
        assert example['clip_id'] == '0011'
        assert example['frame_index'] == 0
        assert example['camera_id'] == 'left'
        assert example['intrinsic_matrix'].shape == (3, 4)
        assert example['extrinsic_matrix'].shape == (4, 4)
        assert example['box'].shape == (4,)
        assert example['crop_box'].shape == (4,)
        assert example['joints_3d'].shape == (17, 3)
        assert example['univ_joints_3d'].shape == (17, 3)
        assert example['skeleton_name'] == 'aspset_17j'
        assert example['joints_2d'].shape == (17, 2)
        assert example['image'].shape == (3, 2160, 3840)

    def test_omit_field(self, aspset_data_path):
        fields = ('cameras', 'boxes', 'joints_3d', 'joints_2d')
        dataset = create_aspset510_dataset(aspset_data_path, 'val', fields=fields)
        example = dataset[0]
        assert 'image' not in example
