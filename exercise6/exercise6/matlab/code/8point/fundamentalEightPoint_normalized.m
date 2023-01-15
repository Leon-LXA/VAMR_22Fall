function F = fundamentalEightPoint_normalized(p1, p2)
% estimateEssentialMatrix_normalized: estimates the essential matrix
% given matching point coordinates, and the camera calibration K
%
% Input: point correspondences
%  - p1(3,N): homogeneous coordinates of 2-D points in image 1
%  - p2(3,N): homogeneous coordinates of 2-D points in image 2
%
% Output:
%  - F(3,3) : fundamental matrix
%
%     N = size(p1,2);
%     Q = zeros(N,9);
%     [p1_tilda, T1] = normalise2dpts(p1);
%     [p2_tilda, T2] = normalise2dpts(p2);
%     for i = 1:N
%         Q(i,:) = kron(p1_tilda(:,i),p2_tilda(:,i)).';
%     end
%     [~,~,V] = svd(Q);
%     F_vec = V(:,9);
%     F = reshape(F_vec, [3,3]);
%     % need to make det(F) = 0, because div(F) = 2
%     [U2,S2,V2] = svd(F);
%     S2(3,3) = 0;
%     F = U2 * S2 * V2';

    [p1_tilda,T1] = normalise2dpts(p1);
    [p2_tilda,T2] = normalise2dpts(p2);

    % Linear solution
    F = fundamentalEightPoint(p1_tilda,p2_tilda);
    
    F = T2' * F * T1;
%     F = F./sqrt(sum(F.^2,'all'));
    
    
end