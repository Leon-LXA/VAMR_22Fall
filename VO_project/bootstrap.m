function [matched_p1, matched_p2, inliers_kpt, E] = bootstrap(img0, img1, K, num_kpts)
    rng(2);
    % find kpts from both imgs using Harris detector
    harris_patch_size = 9;
    harris_kappa = 0.08;
    NMS_radius = 8;
    descriptor_radius = 9;
    match_lambda = 10;

    img0_harris = harris(img0, harris_patch_size, harris_kappa);
    % [h,w], N, int -> kpts [2, N]
    img0_keypoints = selectKeypoints(img0_harris, num_kpts, NMS_radius);
    img0_descriptors = describeKeypoints(img0, img0_keypoints, descriptor_radius);

    img1_harris = harris(img1, harris_patch_size, harris_kappa);
    % [h,w], N, int -> kpts [2, N] in [row; col]
    img1_keypoints = selectKeypoints(img1_harris, num_kpts, NMS_radius);
    img1_descriptors = describeKeypoints(img1, img1_keypoints, descriptor_radius);
    % [(2r+1)^2, N], [(2r+1)^2, N], int -> matches [1, Q]
    matches = matchDescriptors(img1_descriptors, img0_descriptors, match_lambda);
    % both [N, 2]
    matched_p1 = img0_keypoints(:, matches(matches > 0))';
    matched_p2 = img1_keypoints(:, matches > 0)';
    matched_p1 = fliplr(matched_p1);
    matched_p2 = fliplr(matched_p2);
    % figure(1)
    % showMatchedFeatures(img0, img1, matched_p1, matched_p2, "montage");
    % title("Point Matches");

    % estimate F using lib function (with RANSAC)
    thres_estFmat = 1e-2;
    [F_RANSAC, inliers_kpt] = estimateFundamentalMatrix(matched_p1, matched_p2,...
        'Method','RANSAC', 'NumTrials',500, 'DistanceThreshold',thres_estFmat);
    E = K' * F_RANSAC * K;
end