function F = fundamentalEightPoint(p1,p2)
% fundamentalEightPoint  The 8-point algorithm for the estimation of the fundamental matrix F
%
% The eight-point algorithm for the fundamental matrix with a posteriori
% enforcement of the singularity constraint (det(F)=0).
% Does not include data normalization.
%
% Reference: "Multiple View Geometry" (Hartley & Zisserman 2000), Sect. 10.1 page 262.
%
% Input: point correspondences
%  - p1(3,N): homogeneous coordinates of 2-D points in image 1
%  - p2(3,N): homogeneous coordinates of 2-D points in image 2
%
% Output:
%  - F(3,3) : fundamental matrix
    N = size(p1,2);
    Q = zeros(N,9);
    for i = 1:N
        Q(i,:) = kron(p1(:,i),p2(:,i));
    end
    [~,~,V] = svd(Q);
    F_vec = V(:,9);
    F = reshape(F_vec, [3,3]);
    % need to make det(F) = 0, because div(F) = 2
    [U2,S2,V2] = svd(F);
    S2(3,3) = 0;
    F = U2 * S2 * V2';
end