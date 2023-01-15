function [R,T] = disambiguateRelativePose(Rots,u3,points0_h,points1_h,K1,K2)
% DISAMBIGUATERELATIVEPOSE- finds the correct relative camera pose (among
% four possible configurations) by returning the one that yields points
% lying in front of the image plane (with positive depth).
%
% Arguments:
%   Rots -  3x3x2: the two possible rotations returned by decomposeEssentialMatrix
%   u3   -  a 3x1 vector with the translation information returned by decomposeEssentialMatrix
%   p1   -  3xN homogeneous coordinates of point correspondences in image 1
%   p2   -  3xN homogeneous coordinates of point correspondences in image 2
%   K1   -  3x3 calibration matrix for camera 1
%   K2   -  3x3 calibration matrix for camera 2
%
% Returns:
%   R -  3x3 the correct rotation matrix
%   T -  3x1 the correct translation vector
%
%   where [R|t] = T_C2_C1 = T_C2_W is a transformation that maps points
%   from the world coordinate system (identical to the coordinate system of camera 1)
%   to camera 2.
%
    N = size(points0_h,2);

    M1 = K1*[eye(3),[0,0,0]'];
    M2 = K2*[Rots(:,:,1) , u3];
    P1 = linearTriangulation(points0_h,points1_h,M1,M2);
    
    M2 = K2*[Rots(:,:,1) , -u3];
    P2 = linearTriangulation(points0_h,points1_h,M1,M2);
    
    M2 = K2*[Rots(:,:,2) , u3];
    P3 = linearTriangulation(points0_h,points1_h,M1,M2);

    M2 = K2*[Rots(:,:,2) , -u3];
    P4 = linearTriangulation(points0_h,points1_h,M1,M2);
    
    pos_depth_num = zeros(1,4);
    for i = 1:N
        if P1(3,i) > 0
            pos_depth_num(1) = pos_depth_num(1) + 1;
        end
        if P2(3,i) > 0
            pos_depth_num(2) = pos_depth_num(2) + 1;
        end
        if P3(3,i) > 0
            pos_depth_num(3) = pos_depth_num(3) + 1;
        end
        if P4(3,i) > 0
            pos_depth_num(4) = pos_depth_num(4) + 1;
        end
    end
    [~,I] = max(pos_depth_num);
    if I == 1
        R = Rots(:,:,1);
        T = u3;
    elseif I == 2
        R = Rots(:,:,1);
        T = -u3;
    elseif I == 3
        R = Rots(:,:,2);
        T = u3;
    elseif I == 4
        R = Rots(:,:,2);
        T = -u3;
    end
end
