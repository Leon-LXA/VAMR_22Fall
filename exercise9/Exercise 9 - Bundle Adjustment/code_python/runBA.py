import numpy as np
import scipy
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from utils import twist2HomogMatrix
from previous_exercises.projectPoints import projectPoints


def runBA(hidden_state, observations, K):
    """
    Update the hidden state, encoded as explained in the problem statement, with 20 bundle adjustment iterations.
    """
    
