import cv2
import matplotlib.pyplot as plt

from shi_tomasi import shi_tomasi
from harris import harris
from selectKeypoints import selectKeypoints
from describeKeypoints import describeKeypoints
from matchDescriptors import matchDescriptors
from plotMatches import plotMatches


# Randomly chosen parameters that seem to work well - can you find better ones?
corner_patch_size = 9
harris_kappa = 0.08
num_keypoints = 200
nonmaximum_supression_radius = 8
descriptor_radius = 9
match_lambda = 4

img = cv2.imread('../data/000000.png', cv2.IMREAD_GRAYSCALE)

# Part 1 - Calculate Corner Response Functions
# Shi-Tomasi
shi_tomasi_scores = shi_tomasi(img, corner_patch_size)

# Harris
harris_scores = harris(img, corner_patch_size, harris_kappa)
#
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].axis('off')
axs[0, 1].imshow(img, cmap='gray')
axs[0, 1].axis('off')

axs[1, 0].imshow(shi_tomasi_scores)
axs[1, 0].set_title('Shi-Tomasi Scores')
axs[1, 0].axis('off')

axs[1, 1].imshow(harris_scores)
axs[1, 1].set_title('Harris Scores')
axs[1, 1].axis('off')

fig.tight_layout()
plt.show()

# Part 2 - Select keypoints
keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius)

plt.clf()
plt.close()
plt.imshow(img, cmap='gray')
plt.plot(keypoints[1, :], keypoints[0, :], 'rx', linewidth=2)
plt.axis('off')
plt.show()

# Part 3 - Describe keypoints and show 16 strongest keypoint descriptors
descriptors = describeKeypoints(img, keypoints, descriptor_radius)

plt.clf()
plt.close()
fig, axs = plt.subplots(4, 4)
patch_size = 2 * descriptor_radius + 1
for i in range(16):
    axs[i // 4, i % 4].imshow(descriptors[:, i].reshape([patch_size, patch_size]))
    axs[i // 4, i % 4].axis('off')

plt.show()

# Part 4 - Match descriptors between first two images
img_2 = cv2.imread('../data/000001.png', cv2.IMREAD_GRAYSCALE)
harris_scores_2 = harris(img_2, corner_patch_size, harris_kappa)
keypoints_2 = selectKeypoints(harris_scores_2, num_keypoints, nonmaximum_supression_radius)
descriptors_2 = describeKeypoints(img_2, keypoints_2, descriptor_radius)

matches = matchDescriptors(descriptors_2, descriptors, match_lambda)

plt.clf()
plt.close()
plt.imshow(img_2, cmap='gray')
plt.plot(keypoints_2[1, :], keypoints_2[0, :], 'rx', linewidth=2)
plotMatches(matches, keypoints_2, keypoints)
plt.tight_layout()
plt.axis('off')
plt.show()

# Part 5 - Match descriptors between all images
prev_desc = None
prev_kp = None
for i in range(200):
    plt.clf()
    img = cv2.imread('../data/{0:06d}.png'.format(i), cv2.IMREAD_GRAYSCALE)
    scores = harris(img, corner_patch_size, harris_kappa)
    kp = selectKeypoints(scores, num_keypoints, nonmaximum_supression_radius)
    desc = describeKeypoints(img, kp, descriptor_radius)

    plt.imshow(img, cmap='gray')
    plt.plot(kp[1, :], kp[0, :], 'rx', linewidth=2)
    plt.axis('off')

    if prev_desc is not None:
        matches = matchDescriptors(desc, prev_desc, match_lambda)
        plotMatches(matches, kp, prev_kp)
    prev_kp = kp
    prev_desc = desc

    plt.pause(0.1)
