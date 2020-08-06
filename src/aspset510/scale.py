from posekit.skeleton.utils import calculate_knee_neck_height


# The universal scale is taken to be 910 mm along limbs between knee and neck.
UNIV_KNEE_NECK_HEIGHT = 910


def to_univ_scale(joints_3d, skeleton):
    """Scale joints to universal scale, transforming about the origin."""
    k = UNIV_KNEE_NECK_HEIGHT / calculate_knee_neck_height(joints_3d, skeleton.joint_names)
    return joints_3d * k


def to_root_relative_univ_scale(joints_3d, skeleton):
    """Scale joints to universal scale, transforming about the root joint location."""
    root = joints_3d[skeleton.root_joint_id:skeleton.root_joint_id + 1]
    rel_joints_3d = joints_3d - root
    rel_univ_joints_3d = to_univ_scale(rel_joints_3d, skeleton)
    univ_joints_3d = rel_univ_joints_3d + root
    return univ_joints_3d
