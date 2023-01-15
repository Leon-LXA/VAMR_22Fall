function M_tilde = estimatePoseDLT(p, P, K)
% Estimates the pose of a camera using a set of 2D-3D correspondences and K
%
% p: [nx2] vector containing the undistorted coordinates of the 2D points
% P: [nx3] vector containing the 3D point positions
% K: [3x3] camera matrix
% M_tilde=[R_tilde|alpha*t] where R is a rotation matrix. [3x4]
% M_tilde encodes the Trans. that maps points from the W frame to the C frame

    % Convert 2D points to normalized coordinates
    p_normalized = (K \ [p ones(length(p),1)]')';
    %% Build the measurement matrix Q
    num_corners = length(p_normalized);
    Q = zeros(2*num_corners, 12);
    for i=1:num_corners
        Q(2*i-1,1:4) = [P(i,:), 1];
        Q(2*i-1,9:12) = -p_normalized(i,1) * [P(i,:), 1];
        Q(2*i,5:8) = [P(i,:), 1];
        Q(2*i,9:12) = -p_normalized(i,2) * [P(i,:) 1];
    end

    %% Solve for Q * M_tilde = 0 subject to the constraint ||M_tilde||=1
    [~,~,V] = svd(Q);
    M_tilde = V(:, end);
    M_tilde = reshape(M_tilde, 4, 3)';

    %% Extract [R|t] with the correct scale from M_tilde ~ [R|t]
    if det(M_tilde(:,1:3)) < 0
        M_tilde = -M_tilde;
    end
    R = M_tilde(:,1:3);
    % Find the closest orthogonal matrix to R
    [U,~,V] = svd(R);
    R_tilde = U * V';
    % Normalization scheme using the Frobenius norm:
    % recover the unknown scale using the fact that R_tilde is a true rotation matrix
    alpha = norm(R_tilde, 'fro') / norm(R, 'fro');
    % Build M_tilde with the corrected rotation and scale
    M_tilde = [R_tilde alpha * M_tilde(:, 4)];

end

