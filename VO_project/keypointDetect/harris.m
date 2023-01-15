function scores = harris(img, patch_size, kappa)
    % calculate scores of harris detector in each pixel
    [sum_IxIx, sum_IyIy, sum_IxIy] = calculateImgGradient(img, patch_size);
    % calculate corner response
    R = (sum_IxIx .* sum_IyIy) - sum_IxIy .^2 - kappa .* (sum_IxIx + sum_IyIy) .^2;
    patch_radius = (patch_size-1)/2;
    scores = padarray(R, [patch_radius, patch_radius]); %(h * w)
    scores(scores < 0) = 0;
end