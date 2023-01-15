import numpy as np

from linear_triangulation import linearTriangulation

# Number of 3D points to test
N = 10

# Random homogeneous coordinates of 3-D points
P = np.random.rand(4,N)

# Test linear triangulation
P[2, :] = P[2, :] * 5 + 10
P[3, :] = 1


M1 = np.array([ [500,   0,      320,    0],
                [0,     500,    240,    0],
                [0,     0,      1,      0]])

M2 = np.array([ [500,   0,      320,    -100],
                [0,     500,    240,    0],
                [0,     0,      1,      0]])

p1 = M1 @ P
p2 = M2 @ P

P_est = linearTriangulation(p1, p2, M1, M2)

print('P_est - P')
print(P_est - P)
print("Your function looks %s" % ("Correct" if np.allclose(P_est, P) else "Incorrect"))
