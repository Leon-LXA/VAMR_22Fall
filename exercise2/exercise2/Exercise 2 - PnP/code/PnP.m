close all; clc; clear;


A = imread('../data/images_undistorted/img_0001.jpg');
G = rgb2gray(A);
figure(1);
imshow(G);
hold on;
%% initiate data
Pw_list = load('../data/p_W_corners.txt');
K = load('../data/K.txt');
p_list = load('../data/detected_corners.txt');
p_img1 = p_list(1,:);


%% test only 1 pic
M1_tilde = estimatePoseDLT(p_img1, Pw_list, K);
R1_tilde = M1_tilde(:,1:3);
t1_tilde = M1_tilde(:,4);
[U,S,V] = svd(R1_tilde);
R1_approx_tilde = U*V';
alpha = norm(R1_approx_tilde);
R1_approx = R1_approx_tilde/alpha;
t1 = t1_tilde/alpha;
M1 = [R1_approx, t1];

% reproject
reprojected_pts = reprojectPoints(Pw_list, M1, K);

scatter(reprojected_pts(:,1),reprojected_pts(:,2),'filled','r');

M_inv = inv([M1;0,0,0,1]);
plotTrajectory3D(30,M_inv(1:3,4)/100,rotMatrix2Quat(M_inv(1:3,1:3)),Pw_list'/100);
faces=[1 2 4 3];

patch('Faces',faces,'Vertices',Pw_list(1:4,:)/100,'Facecolor','none','LineWidth',1.5,'EdgeColor','r')
patch('Faces',faces,'Vertices',Pw_list(5:8,:)/100,'Facecolor','none','LineWidth',1.5,'EdgeColor','m')
patch('Faces',faces,'Vertices',Pw_list(9:12,:)/100,'Facecolor','none','LineWidth',1.5,'EdgeColor','b')

view([-32,-43])