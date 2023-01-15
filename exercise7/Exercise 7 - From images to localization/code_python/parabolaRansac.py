import numpy as np

def parabolaRansac(data, max_noise):
    """
    best_guess_history is 3xnum_iterations with the polynome coefficients  
    from polyfit of the BEST GUESS SO FAR at each iteration columnwise and max_num_inliers_history is
    1xnum_iterations, with the inlier count of the BEST GUESS SO FAR at each iteration.
    """

