function [pts_tilda, T] = normalise2dpts(pts)
% NORMALISE2DPTS - normalises 2D homogeneous points
%
% Function translates and normalises a set of 2D homogeneous points
% so that their centroid is at the origin and their mean distance from
% the origin is sqrt(2).
%
% Usage:   [pts_tilda, T] = normalise2dpts(pts)
%
% Argument:
%   pts -  3xN array of 2D homogeneous coordinates
%
% Returns:
%   pts_tilda -  3xN array of transformed 2D homogeneous coordinates.
%   T      -  The 3x3 transformation matrix, pts_tilda = T*pts
%
%     pts = pts./(pts(3,:));
    N = size(pts,2);
    
    pts = pts ./ repmat( pts(3,:),3,1);

    % Centroid (Euclidean coordinates)
    mu = mean(pts(1:2,:),2);

    pts_centered = pts(1:2,:)-repmat(mu,1,N);

    sigma = sqrt( mean(sum(pts_centered.^2)) );
    s = sqrt(2) / sigma;

    T = [s,0,-s*mu(1);
         0,s,-s*mu(2);
         0,0,1];
    pts_tilda = T*pts;

end
