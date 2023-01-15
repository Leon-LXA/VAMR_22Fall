function [R_CW, t_CW, best_inmask] ...
    = ransacLocalization(matched_keypoints, landmarks, K)
% keypoints: 2 * n, landmarks: 3 * n
% best_inlier_mask 1 * n, if the match is an outlier, 1 otherwise.
  
%     matched_query_keypoints = flipud(matched_keypoints);
    
    if_adaptive = true;
    % num_iters = 200;
    max_iters = 10000;
    num_sample = 3;
    if(if_adaptive)
        num_iters = inf;
        confidence = 0.90;
    end
    
    pixel_thres = 15;
    best_inliners = 0;
    min_inliners = 10;

    num_match = size(matched_keypoints, 2);
    best_inmask = zeros(1, num_match);
    
    iter = 1;
    while(num_iters > iter)
        % sample from data
        [sample_P, idx] = datasample(landmarks, num_sample, 2, 'Replace', false);
        sample_p = matched_keypoints(:, idx); % [2, k]
        inliners = false(size(sample_p, 2), 1);
        % solve 2D-3D and validate
        % normalize
        p_norm = K \ [sample_p; ones(1, num_sample)]; % [3, k]
        for point = 1:3
            p_norm(:, point) = p_norm(:, point) / norm(p_norm(:, point), 2);
        end
        M_multi = p3p(sample_P, p_norm);
        num_in_solu = 0;
        for solu = 1:4
            R_WC = real(M_multi(:, solu*4 - 2 : solu*4));
            t_WC = real(M_multi(:, solu*4 - 3));
            M_CW = [R_WC', -R_WC' * t_WC];
            p_reproj = reprojectPoints(landmarks', M_CW, K);
            err = sum((p_reproj - matched_keypoints').^2, 2);
            inliners_solu = (err <= pixel_thres^2);
            if(nnz(inliners_solu) > num_in_solu)
                inliners = inliners_solu;
                num_in_solu = nnz(inliners);
                best_M_solu = M_CW;
            end
        end

        % update best inliners
        num_inliners = nnz(inliners);
        if(num_inliners > min_inliners && num_inliners > best_inliners)
            best_inliners = num_inliners;
            best_inmask = inliners;
            best_M = best_M_solu;
        end
        
        % update iterations
        if(if_adaptive)
            outlier_ratio = 1 - best_inliners / num_match;
            % set a upper bound
            outlier_ratio = min(0.9, outlier_ratio);
            num_iters = log(1-confidence) / log(1 - (1-outlier_ratio)^num_sample);
            num_iters = min(max_iters, num_iters);
        end
        iter = iter + 1;
    end
    
    if(best_inliners == 0)
        R_CW = [];
        t_CW = [];
    else
        % recalculate
%         best_M = estimatePoseDLT(...
%         	matched_keypoints(:, best_inmask>0)', landmarks(:, best_inmask>0)', K);

%         pose_error = @(pose) reprojectError(landmarks, matched_keypoints, K* pose, false);
%         options = optimoptions(@lsqnonlin, 'Algorithm', 'levenberg-marquardt',...
%             'MaxIter', 30, 'display', 'off');
%         best_M = lsqnonlin(pose_error, double(best_M), [], [], options);
        R_CW = best_M(:, 1:3);
        t_CW = best_M(:, 4);
    end

end
