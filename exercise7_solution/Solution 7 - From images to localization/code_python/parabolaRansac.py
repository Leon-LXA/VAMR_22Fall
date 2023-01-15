import numpy as np

def parabolaRansac(data, max_noise):
    """
    best_guess_history is 3xnum_iterations with the polynome coefficients  
    from polyfit of the BEST GUESS SO FAR at each iteration columnwise and max_num_inliers_history is
    1xnum_iterations, with the inlier count of the BEST GUESS SO FAR at each iteration.
    """
    num_iterations = 100

    best_guess_history = np.zeros([3, num_iterations])
    max_num_inliers_history = np.zeros([num_iterations])

    best_guess = np.zeros([3, 1])
    max_num_inliers = 0
    rerun_on_inliers = True

    for i in range(num_iterations):
        # Model based on 3 samples:
        indices = np.random.permutation(data.shape[1])[:3]
        samples = data[:, indices]
        guess = np.polyfit(samples[0, :], samples[1,:], 2)

        # Evaluate amount of inliers
        errors = np.abs(np.polyval(guess, data[0, :]) - data[1,:])
        inliers = errors <= max_noise + 1e-5
        num_inliers = (inliers).sum()
        # Determine if the current guess is the best so far.
        if num_inliers > max_num_inliers:
            if rerun_on_inliers:
                guess = np.polyfit(data[0, inliers], data[1, inliers], 2)
            best_guess = guess
            max_num_inliers = num_inliers

        best_guess_history[:, i] = best_guess
        max_num_inliers_history[i] = max_num_inliers

    return best_guess_history, max_num_inliers_history
