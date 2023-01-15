function error = reprojectError(points3d, points2d, KM, is_abs)
%REPROJECTIONERROR compute mean squared euclidean error between 
%                   2d points and projected 3d points
%   points3d - [3xn] coordinates of the 3d points in the world frame
%   points2d - [2xN] corresponding 2d points from the camera
%   cameraMatrix - [3x4] full projection matrix of the camera (K*P)
%   Returns: error - [1xN] array of (non-squared) reprojection errors
    reprojected = (KM * [points3d; ones(1, size(points3d, 2))]);
    reprojected = reprojected(1:2, :) ./ repmat(reprojected(3, :), 2, 1);

    if(is_abs)
        delta_P = sum(reprojected - points2d, 1);
        error = double(delta_P);
    else
        delta_P = sum((reprojected - points2d) .^ 2, 1);
        error = double(sqrt(delta_P));
    end
    
end