import numpy as np

def warpImage(image, W):
    """
    Input
        image   np.ndarray 
        W       2 x 3 np.ndarray

    Output
        warped image of the same dimensions
    """

    warped_image = np.zeros_like(image)

    # minimal and maximal coordinates that are valid in (x,y) style
    min_coords = np.array([0, 0])
    max_coords = image.shape[::-1]
    
    # to achieve a speed comparable to MATLAB, we need to spend some 
    # effort on this function
    xm, ym = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    xm = np.reshape(xm, (1,-1))
    ym = np.reshape(ym, (1,-1))
    pre_warp = np.r_[xm, ym, np.ones_like(xm)]
    warped = (W @ pre_warp).T
    mask = np.logical_or.reduce(np.c_[
        warped[:,0] >= max_coords[0],
        warped[:,0] <= min_coords[0],
        warped[:,1] >= max_coords[1],
        warped[:,1] <= min_coords[1]], axis=1)
    warped[mask,:] = 0
    warped_int = warped.astype('int')
    warped_image = image[warped_int[:,1], warped_int[:,0]]
    warped_image[mask] = 0
    warped_image = np.reshape(warped_image, image.shape)

    # THIS CODE IS IDENTICAL TO THE VECTORIZED IMPLEMENTATION ABOVE
    # BUT ABOUT 1000x FASTER
    #  for x in range(image.shape[1]):
        #  for y in range(image.shape[0]):
            #  warped_pt = (W @ np.array([x, y, 1]).T).T
            #  if np.all(warped_pt < max_coords) and np.all(warped_pt > min_coords):
                #  warped_image[y, x] = image[int(warped_pt[1]), int(warped_pt[0])]
    
    return warped_image

