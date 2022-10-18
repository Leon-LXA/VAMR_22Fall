close all; clc; clear;


for m = 1:210
    A = imread(['../data/images_undistorted/img_',num2str(m,'%04d'),'.jpg']);
    G = rgb2gray(A);
%     figure(1);
%     imshow(G);
%     hold on;
    %% initiate data
    Pw_list = load('../data/p_W_corners.txt');
    K = load('../data/K.txt');
    p_list = load('../data/detected_corners.txt');
    p = p_list(m,:);


    %% test all pic
    M_tilde = estimatePoseDLT(p, Pw_list, K);
    R_tilde = M_tilde(:,1:3);
    t_tilde = M_tilde(:,4);
    [U,S,V] = svd(R_tilde);
    R_approx_tilde = U*V';
    alpha = norm(R_approx_tilde);
    R_approx = R_approx_tilde/alpha;
    t = t_tilde/alpha;
    M = [R_approx, t];

    % reproject
    reprojected_pts = reprojectPoints(Pw_list, M, K);

%     scatter(reprojected_pts(:,1),reprojected_pts(:,2),'filled','r');

    M_inv = inv([M;0,0,0,1]);
%     close all;
    plotTrajectory3D(30,M_inv(1:3,4)/100,rotMatrix2Quat(M_inv(1:3,1:3)),Pw_list'/100);
%     faces=[1 2 4 3];
% 
%     patch('Faces',faces,'Vertices',Pw_list(1:4,:)/100,'Facecolor','none','LineWidth',1.5,'EdgeColor','r')
%     patch('Faces',faces,'Vertices',Pw_list(5:8,:)/100,'Facecolor','none','LineWidth',1.5,'EdgeColor','m')
%     patch('Faces',faces,'Vertices',Pw_list(9:12,:)/100,'Facecolor','none','LineWidth',1.5,'EdgeColor','b')
% 
%     view([-32,-43])
end