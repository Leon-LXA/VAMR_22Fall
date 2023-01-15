import cv2
import numpy as np
import matplotlib.pyplot as plt

# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from estimate_essential_matrix import estimateEssentialMatrix
from decompose_essential_matrix import decomposeEssentialMatrix
from disambiguate_relative_pose import disambiguateRelativePose
from linear_triangulation import linearTriangulation
from draw_camera import drawCamera

img_1 = np.array(cv2.imread('../data/0001.jpg'))
img_2 = np.array(cv2.imread('../data/0002.jpg'))

K = np.array([  [1379.74,   0,          760.35],
                [    0,     1382.08,    503.41],
                [    0,     0,          1 ]] )

# Load outlier-free point correspondences
p1 = np.loadtxt('../data/matches0001.txt')
p2 = np.loadtxt('../data/matches0002.txt')

p1 = np.r_[p1, np.ones((1, p1.shape[1]))]
p2 = np.r_[p2, np.ones((1, p2.shape[1]))]

# Estimate the essential matrix E using the 8-point algorithm
E = estimateEssentialMatrix(p1, p2, K, K);
print("E:\n", E)

# Extract the relative camera positions (R,T) from the essential matrix
# Obtain extrinsic parameters (R,t) from E
Rots, u3 = decomposeEssentialMatrix(E)

# Disambiguate among the four possible configurations
R_C2_W,T_C2_W = disambiguateRelativePose(Rots, u3, p1, p2, K, K)


# Triangulate a point cloud using the final transformation (R,T)
M1 = K @ np.eye(3,4)
M2 = K @ np.c_[R_C2_W, T_C2_W]
P = linearTriangulation(p1, p2, M1, M2)

# Visualize the 3-D scene
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1, projection='3d')

# R,T should encode the pose of camera 2, such that M1 = [I|0] and M2=[R|t]

# P is a [4xN] matrix containing the triangulated point cloud (in
# homogeneous coordinates), given by the function linearTriangulation
ax.scatter(P[0,:], P[1,:], P[2,:], marker = 'o')

# Display camera pose
drawCamera(ax, np.zeros((3,)), np.eye(3), length_scale = 2)
ax.text(-0.1,-0.1,-0.1,"Cam 1")

center_cam2_W = -R_C2_W.T @ T_C2_W
drawCamera(ax, center_cam2_W, R_C2_W.T, length_scale = 2)
ax.text(center_cam2_W[0]-0.1, center_cam2_W[1]-0.1, center_cam2_W[2]-0.1,'Cam 2')

# Display matched points
ax = fig.add_subplot(1,3,2)
ax.imshow(img_1)
ax.scatter(p1[0,:], p1[1,:], color = 'y', marker='s')
ax.set_title("Image 1")

ax = fig.add_subplot(1,3,3)
ax.imshow(img_2)
ax.scatter(p2[0,:], p2[1,:], color = 'y', marker='s')
ax.set_title("Image 2")

plt.show()
