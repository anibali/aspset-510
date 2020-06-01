import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from posekit.skeleton import Skeleton


# Colours to use for groups of joints. Makes it easier to distinguish left from right.
GROUP_COLOURS = dict(
    centre=(1.0, 0.0, 1.0),
    left=(0.0, 0.0, 1.0),
    right=(1.0, 0.0, 0.0),
)


def plot_joints_2d(ax: Axes, joints_2d, skeleton: Skeleton, alpha=1.0, point_size=8, visibilities=None):
    artists = []
    for joint_id, joint in enumerate(joints_2d):
        meta = skeleton.get_joint_metadata(joint_id)
        parent_id = meta['parent']
        color = GROUP_COLOURS[meta['group']]
        if visibilities is not None:
            if not visibilities[joint_id] or not visibilities[parent_id]:
                color = tuple(map(lambda x: max(x, 0.65), color))
        parent = joints_2d[parent_id]
        offset = parent - joint
        if np.linalg.norm(offset, ord=2) >= 1:
            artist = ax.arrow(
                joint[0], joint[1],
                offset[0], offset[1],
                color=color,
                alpha=alpha,
                head_width=2,
                length_includes_head=True,
            )
            artists.append(artist)
    if point_size > 0:
        xs = joints_2d[..., 0:1]
        ys = joints_2d[..., 1:2]
        artists.append(ax.scatter(xs, ys, color='grey', alpha=alpha, s=point_size))
    return artists


def plot_joints_3d(ax: Axes3D, joints_3d, skeleton: Skeleton, invert=True, alpha=1.0, mask=None):
    """Plot a visual representation of the skeleton Matplotlib 3D axes."""
    # NOTE: y and z axes are swapped, but we will relabel them appropriately.
    joints_3d = np.asarray(joints_3d)
    xs = joints_3d[..., 0:1]
    ys = joints_3d[..., 2:3]
    zs = joints_3d[..., 1:2]

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    # Correct aspect ratio (https://stackoverflow.com/a/21765085).
    max_range = np.array([
        xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()
    ]).max() / 2.0
    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if invert:
        ax.invert_zaxis()

    # Set starting view.
    ax.view_init(elev=20, azim=-100)

    artists = []

    # Plot the bones as quivers.
    for joint_id, joint in enumerate(joints_3d):
        meta = skeleton.get_joint_metadata(joint_id)
        color = GROUP_COLOURS[meta['group']]
        parent_id = meta['parent']
        if mask is not None:
            if not mask[joint_id] or not mask[parent_id]:
                color = tuple(map(lambda x: max(x, 0.65), color))
        parent = joints_3d[parent_id]
        offset = parent - joint
        artists.append(ax.quiver(
            [joint[0]], [joint[2]], [joint[1]],
            [offset[0]], [offset[2]], [offset[1]],
            color=color,
            alpha=alpha,
        ))
    # Plot the joints as points.
    artists.append(ax.scatter(xs, ys, zs, color='grey', alpha=alpha))
    return artists
