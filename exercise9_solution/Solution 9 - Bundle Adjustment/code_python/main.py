import numpy as np
import matplotlib.pyplot as plt

# Code from previous exercise
from cropProblem import cropProblem
from utils import twist2HomogMatrix
from alignEstimateToGroundTruth import alignEstimateToGroundTruth
from plotMap import plotMap
from runBA import runBA


hidden_state = np.genfromtxt('../data/hidden_state.txt')
observations = np.genfromtxt('../data/observations.txt')
num_frames = 150
K = np.genfromtxt('../data/K.txt')
poses = np.genfromtxt('../data/poses.txt')
# 'pp' stands for p prime
pp_G_C = poses[:, [3, 7, 11]].T

hidden_state, observations, pp_G_C = cropProblem(hidden_state, observations, pp_G_C, num_frames)
cropped_hidden_state, cropped_observations, _ = cropProblem(hidden_state, observations, pp_G_C, 4)

# Compare trajectory to ground truth.
# Remember, V is the "world frame of the visual odometry"...
T_V_C = hidden_state[:num_frames*6].reshape([-1, 6]).T
p_V_C = np.zeros([3, num_frames])
for i in range(num_frames):
    single_T_V_C = twist2HomogMatrix(T_V_C[:, i])
    p_V_C[:, i] = single_T_V_C[:3, -1]

# ... and G the "world frame of the ground truth".
plt.plot(pp_G_C[-1, :], -pp_G_C[0, :], label='Ground Truth')
plt.plot(p_V_C[-1, :], -p_V_C[0, :], label='Estimate')
plt.axis('equal')
plt.legend()
plt.show()

# Align estimate to ground truth.
p_G_C = alignEstimateToGroundTruth(pp_G_C, p_V_C)

# ... and G the "world frame of the ground truth".
plt.clf()
plt.close()
plt.plot(pp_G_C[-1, :], -pp_G_C[0, :], label='Ground Truth')
plt.plot(p_V_C[-1, :], -p_V_C[0, :], label='Original Estimate')
plt.plot(p_G_C[-1, :], -p_G_C[0, :], label='Aligned Estimate')
plt.axis('equal')
plt.legend()
plt.show()


# Plot the state before bundle adjustment
plotMap(cropped_hidden_state, cropped_observations, [0, 20, -5, 5], title='Cropped problem before bundle adjustment')

# Run BA and plot
cropped_hidden_state = runBA(cropped_hidden_state, cropped_observations, K)
plotMap(cropped_hidden_state, cropped_observations, [0, 20, -5, 5], title='Cropped problem after bundle adjustment')

# Full problem
plotMap(hidden_state, observations, [0, 40, -10, 10], title='Full problem before bundle adjustment')

# Run BA and plot
# Full problems only runs with sparse Jacobian pattern! Additionally, the resulting optimization result for the full
# problem obtained with the scipy least_squares function does not differ much from the original trajectory.
optimized_hidden_state = runBA(hidden_state, observations, K)
plotMap(optimized_hidden_state, observations, [0, 40, -10, 10], title='Full problem after bundle adjustment')

# Verify better performance
T_V_C = optimized_hidden_state[:num_frames*6].reshape([-1, 6]).T
p_V_C = np.zeros([3, num_frames])
for i in range(num_frames):
    single_T_V_C = twist2HomogMatrix(T_V_C[:, i])
    p_V_C[:, i] = single_T_V_C[:3, -1]

p_G_C_optimized = alignEstimateToGroundTruth(pp_G_C, p_V_C)

plt.clf()
plt.close()
plt.plot(pp_G_C[-1, :], -pp_G_C[0, :], label='Ground Truth')
plt.plot(p_G_C[-1, :], -p_G_C[0, :], label='Original Estimate')
plt.plot(p_G_C_optimized[-1, :], -p_G_C_optimized[0, :], label='Optimized Estimate')
plt.axis('equal')
plt.legend()
plt.show()
