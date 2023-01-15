function [new_P, new_X, keep_candidate] =...
    triangulateTrackingPoints(cur_C, cur_T, track_C, track_T, K, angle_thres)
% cur_C [2 ,c]: candidate kpts in current frame
% track_C [2 ,c]: tracked kpts corresponding to candidate kpts
% cur_T [3, 4]: pose of current frame, C_cur to W 
% track_T [12, c]: frame pose corresponding to candidate kpts, C_track to W
    num_candidate = size(cur_C, 2);
    new_P = zeros(2, num_candidate);
    new_X = zeros(3, num_candidate);
    keep_candidate = true(num_candidate, 1);
    
    for c = 1:num_candidate
        M_track = reshape(track_T(:, c), [3, 4]);
        p1 = track_C(:, c);
        p2 = cur_C(:, c);
        % calculate angle
        norm_p1 = K \ [p1; 1];
        norm_p2 = K \ [p2; 1];
        norm_p1 = norm_p1 ./ repmat(norm_p1(3, :), 3, 1);
        norm_p2 = norm_p2 ./ repmat(norm_p2(3, :), 3, 1);
        dir1 =  M_track(:, 1:3) * norm_p1;
        dir2 =  cur_T(:, 1:3) * norm_p2;
        alpha = atan2d(norm(cross(dir1, dir2)), dot(dir1, dir2));
        
        % if the angle between two frame (Tao_i,T_CW) is larger enough, 
        % triangulate the pairs (F_i, C_i) and add the newly P_t and X_t
        if (abs(alpha) >= angle_thres)
            new_landmark = linearTriangulation([p1; 1], [p2; 1], K*M_track, K*cur_T);
            landmark_CAM = cur_T * [new_landmark; 1];
            if(landmark_CAM(3,1) <= 0)
                continue
            end
            new_P(:, c) = p2;
            new_X(:, c) = new_landmark;
            keep_candidate(c, :) = 0;
        end
    end
    new_P = new_P(:, keep_candidate==0);
    new_X = new_X(:, keep_candidate==0);
end

%         new_landmark = linearTriangulation(...
%             [track_C(:, c); 1], [cur_C(:, c); 1], K*M_track, K*cur_T);
%         cur_C_pos = -cur_T(:, 1:3)' * cur_T(:, 4);
%         dir1 = new_landmark(1:3, :) - cur_C_pos;
%         track_C_pos = -M_track(:, 1:3)' * M_track(:, 4);
%         dir2 = new_landmark(1:3, :) - track_C_pos;
%         alpha = atan2d(norm(cross(dir1, dir2)), dot(dir1, dir2));
%         landmark_CAM = cur_T * new_landmark;