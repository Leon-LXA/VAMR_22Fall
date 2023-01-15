%% Setup
close all;clear;
ds = 1; % 0: KITTI, 1: Malaga, 2: parking
kitti_path = 'datasets/kitti';
malaga_path = 'datasets/malaga-urban';
parking_path = 'datasets/parking';

addpath('keypointDetect/')
addpath('poseEstimate/');
addpath('triangulation/');
addpath('KLT/')

% ground_truth: T_WC real pose of the cam in each frame [num_frames, 8]
if ds == 0
    % need to set kitti_path to folder containing "05" and "poses"
    assert(exist('kitti_path', 'var') ~= 0);
    ground_truth = load([kitti_path '/poses/05.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    last_frame = 4540;
    K = [7.188560e+02, 0, 6.071928e+02;
            0, 7.188560e+02, 1.852157e+02;
            0, 0, 1];
elseif ds == 1
    % Path containing the many files of Malaga 7.
    assert(exist('malaga_path', 'var') ~= 0);
    images = dir([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images']);
    left_images = images(3:2:end);
    last_frame = length(left_images);
    K = [621.18428, 0, 404.0076;
            0, 621.18428, 309.05989;
            0, 0, 1];
elseif ds == 2
    % Path containing images, depths and all...
    assert(exist('parking_path', 'var') ~= 0);
    last_frame = 598;
    ground_truth = load([parking_path '/poses.txt']);
    ground_truth = ground_truth(:, [end-8 end]);
    K = [331.37, 0, 320;
            0, 369.568, 240;
            0,      0,   1];
else
    assert(false);
end

%% Bootstrap
% set bootstrap frames with appropriate baseline
if ds == 0
    bootstrap_frames = [1, 3];
    img0 = imread([kitti_path '/05/image_0/' ...
        sprintf('%06d.png', bootstrap_frames(1))]);
    img1 = imread([kitti_path '/05/image_0/' ...
        sprintf('%06d.png', bootstrap_frames(2))]);
elseif ds == 1
    bootstrap_frames = [1, 3];
    img0 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(1)).name]));
    img1 = rgb2gray(imread([malaga_path ...
        '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
        left_images(bootstrap_frames(2)).name]));
elseif ds == 2
    bootstrap_frames = [1, 4];
    img0 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png', bootstrap_frames(1))]));
    img1 = rgb2gray(imread([parking_path ...
        sprintf('/images/img_%05d.png', bootstrap_frames(2))]));
else
    assert(false);
end

rng(2);
% find kpts from both imgs using Harris detector
harris_patch_size = 9;
harris_kappa = 0.08;
num_kpts = 1000;
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
matched_p1 = matched_p1(inliers_kpt, :);
invalid_kpt2 = matched_p2((1-inliers_kpt) > 0, :);
matched_p2 = matched_p2(inliers_kpt, :);

% figure(2)
% showMatchedFeatures(img0, img1, matched_p1, matched_p2, "montage");
% title("Point Matches After Outliers Are Removed");

% decompose E and check possible pose [3, 3] -> [3,3,2], [3,1]
[Rots, u3] = decomposeEssentialMatrix(E);
homo_matched_p1 = [matched_p1';  ones(1, length(matched_p1))];
homo_matched_p2 = [matched_p2';  ones(1, length(matched_p2))];
% [3,3,2], [3,1], [3, N], [3, N], [3,3], [3,3] -> [3,3], [3,1]
% care about the dir of the R & t
[R_C2W, t_C2W] = disambiguateRelativePose(...
    Rots, u3, homo_matched_p1, homo_matched_p2, K, K);

% triangulate the first landmarks from the bootstrap imgs
M_C2W = [R_C2W, t_C2W];
M_WC2 = [R_C2W', -R_C2W' * t_C2W];
disp("First M_CW");
disp(M_C2W);
% [3,N]
first_P = linearTriangulation(...
    homo_matched_p1, homo_matched_p2, K *eye(3, 4), K *M_C2W);
valid_P = first_P(3, :) > 0;
first_P = first_P(:, valid_P);
matched_p2 = matched_p2(valid_P, :);

figure(3)
imshow(img1); hold on;
plot(matched_p2(:, 1), matched_p2(: ,2), 'gx', 'Linewidth', 2);
plot(invalid_kpt2(:, 1), invalid_kpt2(: ,2), 'rx', 'Linewidth', 1);hold off;
pause(0.01);

%% Continuous operation
% setup the initial state S_0
% State - Struct includes: P_i, X_i, C_i, F_i, Tao_i
% P_i [2, n] - kpt in the i-th frame that has correspongding landmark X_i [3, n]
% C_i [2, m] - kpt that doesn't match a landmark
% F_i [2, m]- vaild tracked kpt, each from the fist frame they occur, corresponging to a kpt in C_i
% Tao_i [12, m] - cam pose in the the first frame the tracking kpt occur
prev_img = img1;
P_prev = matched_p2'; X_prev = first_P;
C_prev = invalid_kpt2'; F_prev = invalid_kpt2';
Tao_prev = repmat(reshape(M_C2W, [12, 1]), 1, size(C_prev, 2));
% looping
range = (bootstrap_frames(2) + 1) : 50; %last_frame;
% history array
num_landmarks_hist = zeros(length(range)+1);
poseWC_hist = zeros(12, length(range) + 1);
num_landmarks_hist(1) = size(X_prev, 2);
poseWC_hist(:, 1) = reshape(M_WC2, [12, 1]);

for i = range
    fprintf('\n=========== Processing frame %d ==========\n', i);
    % get next frame
    if ds == 0
        img = imread([kitti_path '/05/image_0/' sprintf('%06d.png', i)]);
    elseif ds == 1
        img = rgb2gray(imread([malaga_path '/malaga-urban-dataset-extract-07_rectified_800x600_Images/' ...
            left_images(i).name]));
    elseif ds == 2
        img = im2uint8(rgb2gray(imread([parking_path sprintf('/images/img_%05d.png', i)])));
    else
        assert(false);
    end
    
    % use P_i-1 to track P_i by KLT
    kptTracker = vision.PointTracker(...
        'NumPyramidLevels', 5, 'MaxBidirectionalError', 0.3,...
        'BlockSize', [31, 31], 'MaxIterations', 40);  
    initialize(kptTracker, P_prev', prev_img);   
    [P_i, keep_KLT] = step(kptTracker, img);
    P_i = round(P_i)';
%     patch_radius = 15;
%     num_iters_KLT = 40;
%     thres_KLT = 2;
%     delta_kpt_P = zeros(size(P_prev));
%     keep_KLT = true(1, size(P_prev, 2));
%     % parallel loop for all kpts
%     parfor k = 1:size(P_prev, 2)
%         [delta_kpt_P(:, k), keep_KLT(k)] = trackKLTRobustly(...
%             prev_img, img, P_prev(:, k)', patch_radius, num_iters_KLT, thres_KLT);
%     end
%     P_i = round(P_prev + delta_kpt_P);
    % candidates from KLT [2, c1]
    C_KLT = P_i(:, ~keep_KLT);
    P_i = P_i(:, keep_KLT);
    P_prev = P_prev(:, keep_KLT);
    X_i = X_prev(:, keep_KLT);
    fprintf('Keypoints success tracked: %d\n', nnz(keep_KLT));
    
    % use P_i and X_i to estimate cam pose T_CW = [R_CW | t_CW] in the current frame
    % record the P_i and correspongding X_i-1(X_i) that pass the RANSAC
    % for kpt not passing the test, store them in C_t
    [R_CnewW, t_CW, inlier_mask] = ransacLocalization(P_i, X_i, K);
    M_frame = [R_CnewW, t_CW];
    disp("New M_CW");
    disp(M_frame);
    M_frame_WC = [R_CnewW', -R_CnewW' *  t_CW];
    % candidates from Localization [2, c2]
    C_localize = P_i(:, (1 - inlier_mask) > 0);
    fprintf('Keypoints failed to track: %d\n', nnz(1 - keep_KLT));
    fprintf('Keypoints failed to localize: %d\n', nnz(1 - inlier_mask));
    
%     figure(4)
%     showMatchedFeatures(prev_img, img, P_prev(:, inlier_mask)', P_i(:, inlier_mask)', "montage");
%     pause(0.01);
    
    figure(5)
    imshow(img); hold on;
    plot(P_i(1, inlier_mask), P_i(2, inlier_mask), 'gx', 'Linewidth', 2);
    
    % use KLT to track kpt in C_i-1, process triangulate check in each tracked kpt in frame i
    candidatekptTracker = vision.PointTracker(...
        'NumPyramidLevels', 5, 'MaxBidirectionalError', 0.3,...
        'BlockSize', [31, 31], 'MaxIterations', 40);
    initialize(candidatekptTracker, C_prev', prev_img);   
    [C_i, keep_KLT] = step(candidatekptTracker, img);
    C_i = round(C_i)';
%     thres_KLT = 0.6;
%     delta_kpt_C = zeros(size(C_prev));
%     keep_KLT = true(1, size(C_prev, 2));
%     parfor k = 1:size(C_prev, 2)
%         [delta_kpt_C(:, k), keep_KLT(k)] = trackKLTRobustly(...
%             prev_img, img, C_prev(:, k)', patch_radius, num_iters_KLT, thres_KLT);
%     end
%     C_i = round(C_prev + delta_kpt_C);
    % discard candidate points that failed to track
    C_i = C_i(:, keep_KLT);
    F_i = F_prev(:, keep_KLT);
    Tao_i = Tao_prev(:, keep_KLT);
    fprintf('Candidate kpts success tracked: %d\n', nnz(keep_KLT));
%     fprintf('Candidate kpts discarded: %d\n', nnz(1 - keep_KLT));
    
    % if the angle between two frame (Tao_i,T_CW) is larger enough, 
    % triangulate the pairs (F_i, C_i) and add the newly P_t and X_t
    alpha_thres = 1;
    % [2, p], [3, p], [q]
    [new_P, new_X, keep_candidate] =...
        triangulateTrackingPoints(C_i, M_frame, F_i, Tao_i, K, alpha_thres);
    P_i = [P_i(:, inlier_mask), new_P];
    X_i = [X_i(:, inlier_mask), new_X];
    
    plot(new_P(1, :), new_P(2, :), 'yx', 'Linewidth', 2);
    fprintf('New 2D-3D pairs triangulated: %d\n', size(new_P, 2));
    
    % update the newly occur candidate kpt C_t into C_i and F_i, as well as Tao_i
    % take care about rudundant!!!!!
    C_i = [C_i(:, keep_candidate), C_KLT, C_localize];
    F_i = [F_i(:, keep_candidate), C_KLT, C_localize];
    num_new_candidate = size(C_KLT, 2) + size(C_localize, 2);
    frame_pose = reshape(M_frame, [12, 1]);
    Tao_i = [Tao_i(:, keep_candidate), repmat(frame_pose, 1, num_new_candidate)];
    fprintf('Current candidate number: %d\n', size(C_i, 2));
    
    plot(C_i(1, :), C_i(2, :), 'cx', 'Linewidth', 1);
    plot(C_KLT(1, :), C_KLT(2, :), 'rx', 'Linewidth', 1.2);
    plot(C_localize(1, :), C_localize(2, :), 'mx', 'Linewidth', 1.2);
    hold off;
    pause(0.01);

    % update state
    prev_img = img;
    P_prev = P_i; X_prev = X_i;
    C_prev = C_i; F_prev = F_i; Tao_prev = Tao_i;

    % save log
    iter = i - bootstrap_frames(2);
    num_landmarks_hist(iter+1) = size(X_i, 2);
    poseWC_hist(:, iter + 1) = reshape(M_frame_WC, [12,1]);
    
    % display 
    figure(6)
    displayTracking(img, iter, P_i, X_i, C_i, num_landmarks_hist, poseWC_hist);
end
