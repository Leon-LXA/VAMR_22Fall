import cv2
import warnings
import numpy as np
import matplotlib.pyplot as plt

from getDisparity import getDisparity
from disparityToPointCloud import disparityToPointCloud

# Scaling down by a factor of 2, otherwise too slow.
left_img = cv2.resize(cv2.imread('../data/left/000000.png', cv2.IMREAD_GRAYSCALE), None,
                      fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
right_img = cv2.resize(cv2.imread('../data/right/000000.png', cv2.IMREAD_GRAYSCALE), None,
                       fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
K = np.loadtxt('../data/K.txt')
K[0:2, :] /= 2

poses = np.loadtxt('../data/poses.txt')

# Given by the KITTI dataset:
baseline = 0.54

# Carefully tuned by the TAs:
patch_radius = 5
min_disp = 5
max_disp = 50
xlims = [7, 20]
ylims = [-6, 10]
zlims = [-5, 5]

# Parts 1, 2 and 4: Disparity on one image pair
disp_img = getDisparity(left_img, right_img, patch_radius, min_disp, max_disp)

plt.clf()
plt.close()
plt.imshow(disp_img)
plt.axis('off')
plt.tight_layout()
plt.show()

# Optional (only if fast enough): Disparity movie
warnings.warn('Visualizing disparity over sequence! This is optional and could take a lot of time (100 stereo pairs)!')

maxi = 99
for i in range(maxi):
    l = cv2.resize(cv2.imread('../data/left/{0:06d}.png'.format(i), cv2.IMREAD_GRAYSCALE), None,
                   fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    r = cv2.resize(cv2.imread('../data/right/{0:06d}.png'.format(i), cv2.IMREAD_GRAYSCALE), None,
                   fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    disp_img_i = getDisparity(l, r, patch_radius, min_disp, max_disp)

    plt.clf()
    plt.imshow(disp_img_i)
    plt.axis('off')
    plt.tight_layout()

    plt.pause(0.1)

# Part 3: Create point cloud for first pair
p_C_points, intensities = disparityToPointCloud(disp_img, K, baseline, left_img)
T_C_F = np.asarray([[0, -1, 0],
                    [0, 0, -1],
                    [1, 0, 0]])
p_F_points = np.matmul(np.linalg.inv(T_C_F), p_C_points[::11, :, None]).squeeze(-1)
intensities = intensities[::11]
intensities = np.tile(intensities[:, None], (1, 3)) / 255

plt.clf()
plt.close()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(p_F_points[:, 0], p_F_points[:, 1], p_F_points[:, 2], s=1, c=intensities)
ax.set_box_aspect((1, 1, 12/40))

ax.azim = 120
ax.dist = 7
ax.elev = 20

ax.axes.set_xlim3d(left=0, right=40)
ax.axes.set_ylim3d(bottom=-20, top=20)
ax.axes.set_zlim3d(bottom=-2, top=10)

plt.show()

# Visualize point clouds over sequence
warnings.warn('Visualizing pointcloud over sequence! This is optional and could take a lot of time (100 stereo pairs)!')

maxi = 99
for i in range(maxi):
    l = cv2.resize(cv2.imread('../data/left/{0:06d}.png'.format(i), cv2.IMREAD_GRAYSCALE), None,
                   fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    r = cv2.resize(cv2.imread('../data/right/{0:06d}.png'.format(i), cv2.IMREAD_GRAYSCALE), None,
                   fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    disp_img_i = getDisparity(l, r, patch_radius, min_disp, max_disp)

    p_C_points, intensities = disparityToPointCloud(disp_img_i, K, baseline, l)
    T_C_F = np.asarray([[0, -1, 0],
                        [0, 0, -1],
                        [1, 0, 0]])
    p_F_points = np.matmul(np.linalg.inv(T_C_F), p_C_points[::11, :, None]).squeeze(-1)
    intensities = intensities[::11]
    intensities = np.tile(intensities[:, None], (1, 3)) / 255

    plt.clf()
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(p_F_points[:, 0], p_F_points[:, 1], p_F_points[:, 2], s=1, c=intensities)
    ax.set_box_aspect((1, 1, 12/40))

    ax.azim = 120
    ax.dist = 7
    ax.elev = 20

    ax.axes.set_xlim3d(left=0, right=40)
    ax.axes.set_ylim3d(bottom=-20, top=20)
    ax.axes.set_zlim3d(bottom=-2, top=10)

    plt.show()
