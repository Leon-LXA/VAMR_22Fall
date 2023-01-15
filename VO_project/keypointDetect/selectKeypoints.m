function keypoints = selectKeypoints(scores, num, r)
% Selects the num best scores as keypoints and performs non-maximum 
% supression of a (2r + 1)*(2r + 1) box around the current maximum.
% return - coordinate of keypoints size:(2, num)
    keypoints = zeros(2, num);
    scores = padarray(scores, [r r]);
    for i = 1:num
        [~, kp] = max(scores(:));
        [row, col] = ind2sub(size(scores), kp);
        kp = [row; col];
        keypoints(:, i) = kp - r;
        scores(kp(1)-r : kp(1)+r, kp(2)-r : kp(2)+r) = zeros(2 * r + 1, 2 * r + 1);
    end
end
