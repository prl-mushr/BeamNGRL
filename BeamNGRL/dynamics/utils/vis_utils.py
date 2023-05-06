import numpy as np
from .dataset_utils import to_np, project_traj_to_map
from PIL import Image
import cv2


def make_heatmap(bev_map, cmap='inferno'):
    import matplotlib.cm
    cmap = matplotlib.cm.get_cmap(cmap)
    bev_map = bev_map / 255.0
    color = cmap(bev_map)[:, :, :3] * 255.0
    ret = color.astype(np.uint8)
    return ret


def project_traj_to_img(traj, img, color=(255, 0, 0), type=None, size=2):
    for i in range(traj.shape[0]):
        if type == 'circle':
            cv2.circle(img, tuple(traj[i, :2]), size, color, -1, cv2.LINE_AA)
        elif type =='square':
            tl = traj[i, :2] + np.array([-size, size])
            br = traj[i, :2] + np.array([size, -size])
            cv2.rectangle(img, tl, br, color, -1)
        else:
            x, y = traj[i, :2]
            img[y, x, :] = color
    return img


def visualize_bev_traj(state, future_traj, past_traj, bev_map, resolution):

    state = to_np(state)
    future_traj = to_np(future_traj)
    past_traj = to_np(past_traj)
    bev_map = to_np(bev_map)

    height, width, n_channels = bev_map.shape
    if n_channels == 3:
        bev_img = bev_map.astype(np.uint8)
    elif n_channels == 1:
        bev_img = make_heatmap(bev_map)
    else:
        raise IOError('BEV map contains an invalid number of channels '
                      'for visualization.')

    future_traj_bev, _ = project_traj_to_map(future_traj, height, resolution)
    past_traj_bev, _ = project_traj_to_map(past_traj, height, resolution)
    state_bev, _ = project_traj_to_map(state[None, :], height, resolution)

    # Plot trajectories
    size = 5
    bev_img = project_traj_to_img(future_traj_bev, bev_img, [0, 0, 255], 'circle', size)
    bev_img = project_traj_to_img(past_traj_bev, bev_img, [255, 0, 0], 'circle', size)
    bev_img = project_traj_to_img(state_bev, bev_img, [0, 255, 0], 'circle', size)

    final_img = Image.fromarray(bev_img.astype(np.uint8))

    return final_img
