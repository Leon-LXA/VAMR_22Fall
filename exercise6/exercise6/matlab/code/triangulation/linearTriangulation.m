function P = linearTriangulation(p1,p2,M1,M2)
% LINEARTRIANGULATION  Linear Triangulation
%
% Input:
%  - p1(3,N): homogeneous coordinates of points in image 1
%  - p2(3,N): homogeneous coordinates of points in image 2
%  - M1(3,4): projection matrix corresponding to first image
%  - M2(3,4): projection matrix corresponding to second image
%
% Output:
%  - P(4,N): homogeneous coordinates of 3-D points
    
    N = size(p1,2);
    P = zeros(4,N);
    for i = 1:N
        p1_cross = [0, -p1(3,i), p1(2,i);
                    p1(3,i), 0, -p1(1,i);
                    -p1(2,i), p1(1,i), 0];
        p2_cross = [0, -p2(3,i), p2(2,i);
                    p2(3,i), 0, -p2(1,i);
                    -p2(2,i), p2(1,i), 0];
        A = [p1_cross*M1;
             p2_cross*M2];
        [~,~,V] = svd(A);
        p_vec = V(:,4);
        P(:,i) = p_vec/p_vec(4);
    end

end