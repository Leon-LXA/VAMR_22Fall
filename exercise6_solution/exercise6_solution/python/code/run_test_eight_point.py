import numpy as np

from fundamental_eight_point import fundamentalEightPoint
from fundamental_eight_point_normalized import fundamentalEightPointNormalized
from utils import distPoint2EpipolarLine

# Number of 3D points to test
N = 40

# Random homogeneous coordinates of 3-D points
#  X = np.random.rand(4,N)
X = np.loadtxt("matlab_X.csv", delimiter = ",")

# Simulated scene with error-free correspondances
X[2, :] = X[2, :] * 5 + 10
X[3, :] = 1

P1 = np.array([ [500,   0,      320,    0],
                [0,     500,    240,    0],
                [0,     0,      1,      0]])

P2 = np.array([ [500,   0,      320,    -100],
                [0,     500,    240,    0],
                [0,     0,      1,      0]])

# Image (i.e. projected points)
x1 = P1 @ X
x2 = P2 @ X


sigma = 1e-1
#  noisy_x1 = x1 + sigma * np.random.randn(*x1.shape)
#  noisy_x2 = x2 + sigma * np.random.randn(*x2.shape)

# If you want to get the same results as matlab users, uncomment those two lines
noisy_x1 = np.loadtxt("matlab_noisy_x1.csv", delimiter = ",")
noisy_x2 = np.loadtxt("matlab_noisy_x2.csv", delimiter = ",")

# Estimate Fundamental Matrix via 8-point algorithm
F  = fundamentalEightPoint(x1, x2)
cost_algebraic = np.linalg.norm( np.sum(x2 * (F @ x1)) ) / np.sqrt(N)
cost_dist_epi_line = distPoint2EpipolarLine(F, x1, x2)

print("")
print('Noise-free correspondences');
print('Algebraic error: %f' % cost_algebraic);
print('Geometric error: %f px' % cost_dist_epi_line);

# Test with noise
F  = fundamentalEightPoint(noisy_x1, noisy_x2) # This gives bad results!

cost_algebraic = np.linalg.norm( np.sum(noisy_x2 * (F @ noisy_x1), axis=0) ) / np.sqrt(N)
cost_dist_epi_line = distPoint2EpipolarLine(F, noisy_x1, noisy_x2)

print("")
print('Noisy correspondences with 8 Point Algorithm')
print('Algebraic error: %f' % cost_algebraic)
print('Geometric error: %f px' % cost_dist_epi_line)

# Test with noise
#  F  = fundamentalEightPoint(noisy_x1, noisy_x2) # This gives bad results!
F  = fundamentalEightPointNormalized(noisy_x1, noisy_x2) # This gives good results!

cost_algebraic = np.linalg.norm( np.sum(noisy_x2 * (F @ noisy_x1), axis=0) ) / np.sqrt(N)
cost_dist_epi_line = distPoint2EpipolarLine(F, noisy_x1, noisy_x2)

print("")
print('Noisy correspondences with normalized 8 Point Algorithm')
print('Algebraic error: %f' % cost_algebraic)
print('Geometric error: %f px' % cost_dist_epi_line)
