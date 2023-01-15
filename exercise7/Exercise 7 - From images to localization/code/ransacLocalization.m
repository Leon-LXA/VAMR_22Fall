function [R_C_W, t_C_W, best_inlier_mask, max_num_inliers_history, num_iteration_history] ...
    = ransacLocalization(matched_query_keypoints, corresponding_landmarks, K)
% query_keypoints should be 2x1000
% all_matches should be 1x1000 and correspond to the output from the
%   matchDescriptors() function from exercise 3.
% best_inlier_mask should be 1xnum_matched (!!!) and contain, only for the
%   matched keypoints (!!!), 0 if the match is an outlier, 1 otherwise.
rng(2)
max_noise = 10;
num_match = size(matched_query_keypoints,2);
matched_query_keypoints_new = [matched_query_keypoints(2,:);matched_query_keypoints(1,:)];
matched_query_keypoints = matched_query_keypoints_new;
iter_num = 2000;
max_num_inliers_history = zeros(1,iter_num);
num_iteration_history = zeros(1, iter_num);
max_num_inliers = -1;

for i = 1:iter_num
    [sample_pt,idx] = datasample(matched_query_keypoints',12,'Replace',false); % 12*2
    landmark = corresponding_landmarks';
    Pw_list = landmark(idx,:); % 12 * 3
    p_img1 = reshape(sample_pt',1,24);
    
    M1_tilde = estimatePoseDLT(p_img1, Pw_list, K);
    R1_tilde = M1_tilde(:,1:3);
    t1_tilde = M1_tilde(:,4);
    [U,~,V] = svd(R1_tilde);
    R1_approx_tilde = U*V';
    alpha = norm(R1_approx_tilde);
    R1_approx = R1_approx_tilde/alpha;
    t1 = t1_tilde/alpha;
    M1 = [R1_approx, t1];
   
    
    reprojected_pts = reprojectPoints(landmark, M1, K); % N*2
    reprojected_tran = reprojected_pts';
    uv_delta = abs(reprojected_tran - matched_query_keypoints); % 2 * N
    uv_delta_max = max(uv_delta);
    k = find(uv_delta_max < max_noise);
    num_inliers = size(k,2);
%     if num_inliers==18
%         num_inliers
%     end
    
    if num_inliers > max_num_inliers
        max_num_inliers = num_inliers;
        max_num_inliers_history(i) = max_num_inliers;
        R_C_W = R1_approx;
        t_C_W = t1;
        best_inlier_mask = zeros(1,num_match);
        best_inlier_mask(1,k) = 1;
        num_iteration_history(i) = log(1-0.99)/log(1-(num_inliers/num_match)^12);
        if num_iteration_history(i) < 0
            num_iteration_history(i) = -num_iteration_history(i);
        end
    else
        max_num_inliers_history(i) = max_num_inliers_history(i-1);
        num_iteration_history(i) = num_iteration_history(i-1);
    end
end

iter_num = floor(num_iteration_history(iter_num));
if iter_num > 30000
    iter_num = 30000;
end
if iter_num <= 2000
    iter_num = 2001;
end
max_num_inliers_history = [max_num_inliers_history,zeros(1,iter_num-40)];
num_iteration_history = [num_iteration_history,zeros(1, iter_num-40)];

for i = 2001:iter_num
    [sample_pt,idx] = datasample(matched_query_keypoints',12,'Replace',false); % 12*2
    landmark = corresponding_landmarks';
    Pw_list = landmark(idx,:); % 12 * 3
    p_img1 = reshape(sample_pt',1,24);
    
    M1_tilde = estimatePoseDLT(p_img1, Pw_list, K);
    R1_tilde = M1_tilde(:,1:3);
    t1_tilde = M1_tilde(:,4);
    [U,~,V] = svd(R1_tilde);
    R1_approx_tilde = U*V';
    alpha = norm(R1_approx_tilde);
    R1_approx = R1_approx_tilde/alpha;
    t1 = t1_tilde/alpha;
    M1 = [R1_approx, t1];
   
    
    reprojected_pts = reprojectPoints(landmark, M1, K); % N*2
    reprojected_tran = reprojected_pts';
    uv_delta = abs(reprojected_tran - matched_query_keypoints); % 2 * N
    uv_delta_max = max(uv_delta);
    k = find(uv_delta_max < max_noise);
    num_inliers = size(k,2);
%     if num_inliers==18
%         num_inliers
%     end
    
    if num_inliers > max_num_inliers
        max_num_inliers = num_inliers;
        max_num_inliers_history(i) = max_num_inliers;
        R_C_W = R1_approx;
        t_C_W = t1;
        best_inlier_mask = zeros(1,num_match);
        best_inlier_mask(1,k) = 1;
        num_iteration_history(i) = log(1-0.99)/log(1-(num_inliers/num_match)^12);
    else
        max_num_inliers_history(i) = max_num_inliers_history(i-1);
        num_iteration_history(i) = num_iteration_history(i-1);
    end
end



end