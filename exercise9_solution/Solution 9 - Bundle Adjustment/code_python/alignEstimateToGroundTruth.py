import numpy as np
from scipy.optimize import least_squares

from utils import HomogMatrix2twist, twist2HomogMatrix


def alignEstimateToGroundTruth(pp_G_C, p_V_C):
    """
    Returns the points of the estimated trajectory p_V_C transformed into the ground truth frame G.
    The similarity transform Sim_G_V is to be chosen such that it results in the lowest error between
    the aligned trajectory points p_G_C and the points of the ground truth trajectory pp_G_C.
    All matrices are 3xN
    """
    # Initial guess is identity.
    twist_guess = HomogMatrix2twist(np.eye(4))
    scale_guess = 1

    def alignError(x):
        T_G_V = twist2HomogMatrix(x[:6])
        scale_G_V = x[6]
        p_G_C = np.matmul(scale_G_V * T_G_V[None, :3, :3], p_V_C.T[:, :, None]).squeeze(-1) + T_G_V[None, :3, 3]
        errors = pp_G_C.T - p_G_C

        return errors.flatten()

    x0 = np.concatenate([twist_guess, np.array([scale_guess])])
    res_1 = least_squares(alignError, x0)
    x_optim = res_1.x

    T_G_V = twist2HomogMatrix(x_optim[:6])
    scale_G_V = x_optim[6]

    p_G_C = np.matmul(scale_G_V * T_G_V[None, :3, :3], p_V_C.T[:, :, None]).squeeze(-1) + T_G_V[None, :3, 3]

    return p_G_C.T
