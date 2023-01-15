clc, clear;

%% videw setup
workingDir = tempname;
mkdir(workingDir);
mkdir(workingDir,'images');
shuttleVideo = VideoReader('shuttle.avi');

%% Read Image and matrix
A = imread('./data/images/img_0001.jpg');
% imshow(A);
G = rgb2gray(A);
figure(1);
imshow(G);
hold on;

k1 = -1.6774e-06;
k2 = 2.5847e-12;

K = [420.506712, 0,          355.208298;
    0,         420.610940,  250.336787;
    0,         0,          1.];

pose_list = load('./data/poses.txt');
% pose = [-0.372483192214,0.0397022486165,0.0650393402332,-0.107035863625,-0.147065242923,0.398512498053];

%% corner pts pos
xw = 0:0.04:0.32;
yw = 0:0.04:0.20;
[Xw, Yw] = meshgrid(xw,yw);

%% get Pp on distorted img
T = poseVectorToTransformationMatrix(pose_list(1,:));
for i=1:54
    Pp = projectPoints(T,K,[Xw(i),Yw(i),0]);
    r_sqr = (Pp(1)-K(1,3))^2+(Pp(2)-K(2,3))^2;
    distort_Pp = (1+k1* r_sqr+ k2 * r_sqr^2)*[Pp(1)-K(1,3);Pp(2)-K(2,3)] + [K(1,3);K(2,3)];
    scatter(distort_Pp(1),distort_Pp(2),'filled','r')
    hold on;
end

%% undistortion
u0 = K(1,3); v0 = K(2,3);

% nearest neighbor VECTOR version
% [X,Y] = meshgrid(0:size(G, 2)-1, 0:size(G, 1)-1);
% px_locs = [X, Y];
% X_diff = (X - u0).^2;
% Y_diff = (Y - v0).^2;
% dist_px_locs = 1 + k1 * (X_diff+Y_diff) + k2 * (X_diff + Y_diff).^2;
% mat_num_x =round(dist_px_locs .* (X - u0) + u0);  
% mat_num_y =round(dist_px_locs .* (Y - v0) + v0);
% for i = 1:size(G,1)
%     for j = 1:size(G,2)
%         undis(i,j) = G(mat_num_y(i,j),mat_num_x(i,j));
%     end
% end

% nearest neighbor FOR version
% for i = 1:size(G,1)
%     for j = 1:size(G,2)
%         r_sqr = (i-v0)^2+(j-u0)^2;
%         mat_num_dist(i,j,:) = round((1+k1* r_sqr+ k2 * r_sqr^2)*[i-v0;j-u0] + [v0;u0]);
%     end
% end
% for i = 1:size(G,1)
%     for j = 1:size(G,2)
%         undis(i,j) = G(mat_num_dist(i,j,1),mat_num_dist(i,j,2));
%     end
% end

% bilinear interpolation FOR version
for i = 1:size(G,1)
    for j = 1:size(G,2)
        r_sqr = (i-v0)^2+(j-u0)^2;
        XYd(:) = (1+k1* r_sqr+ k2 * r_sqr^2)*[i-v0;j-u0] + [v0;u0];
        Xd = XYd(1); Yd = XYd(2);
        X1 = round(Xd);
        Y1 = round(Yd);
        X2 = X1+1; Y2 = Y1+1;
        nni_val = double([G(X1,Y1),G(X1,Y2),G(X2,Y1),G(X2,Y2)]);
        undis(i,j) = uint8((1/(X2-X1)/(Y2-Y1))*nni_val* ...
                    [X2*Y2,-Y2,-X2,1;
                     -X2*Y1,Y1,X2,-1;
                     -X1*Y2,Y2,X1,-1;
                     X1*Y1,-Y1,-X1,1]*[1;Xd;Yd;Xd*Yd]);
    end
end
figure(2)
imshow(undis)
hold on;

for i=1:54
    Pp = projectPoints(T,K,[Xw(i),Yw(i),0]);
    scatter(Pp(1),Pp(2),'filled','r')
    hold on;
end