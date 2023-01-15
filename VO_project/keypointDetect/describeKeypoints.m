function descriptors = describeKeypoints(img, keypoints, r)
% Returns a (2r+1)^2xN matrix of image patch vectors
% keypoints: 2xN matrix containing the keypoint coordinates.
% r is the patch "radius".
    patch_size = 2*r + 1;
    k_size = size(keypoints, 2);
    descriptors = zeros(patch_size^2, k_size);
    img = padarray(img, [r, r]);
    for i = 1:k_size
        posX = keypoints(1, i) + r;
        posY = keypoints(2, i) + r;
        descriptors(:, i) = reshape(img(posX-r : posX+r, posY-r : posY+r), [patch_size^2, 1]);
    end
end
