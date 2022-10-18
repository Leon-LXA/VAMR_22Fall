clc, clear;

%% videw setup
workingDir = tempname;
mkdir(workingDir);
mkdir(workingDir,'images');
shuttleVideo = VideoReader('shuttle.avi');

%% Read Image and matrix

for m = 1:60
    m
    A = imread(['data/images/img_',num2str(m,'%04d'),'.jpg']);
    % imshow(A);
    G = rgb2gray(A);
    figure(1);
%     set(gcf,'Position',[0 0 480 752]);
%     imshow(G);
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
    T = poseVectorToTransformationMatrix(pose_list(m,:));
%     for i=1:54
%         Pp = projectPoints(T,K,[Xw(i),Yw(i),0]);
%         r_sqr = (Pp(1)-K(1,3))^2+(Pp(2)-K(2,3))^2;
%         new_Pp = (1+k1* r_sqr+ k2 * r_sqr^2)*[Pp(1)-K(1,3);Pp(2)-K(2,3)] + [K(1,3);K(2,3)];
%         scatter(new_Pp(1),new_Pp(2),'filled','r')
%         hold on;
%     end

    u0 = K(1,3); v0 = K(2,3);
    for i = 1:size(G,1)
        for j = 1:size(G,2)
    %         mat_num_ori(i,j,1) = i;
    %         mat_num_ori(i,j,2) = j;
            r_sqr = (i-v0)^2+(j-u0)^2;
            mat_num_dist(i,j,:) = round((1+k1* r_sqr+ k2 * r_sqr^2)*[i-v0;j-u0] + [v0;u0]);
        end
    end

    %% undistort
    for i = 1:size(G,1)
        for j = 1:size(G,2)
            r_sqr = (i-v0)^2+(j-u0)^2;
            XYd(:) = (1+k1* r_sqr+ k2 * r_sqr^2)*[i-v0;j-u0] + [v0;u0];
            Xd = XYd(1); Yd = XYd(2);
            X1 = round(Xd);
            Y1 = round(Yd);
            X2 = X1+1; Y2 = Y1+1;
            nni_val = double([G(X1,Y1),G(X1,Y2),G(X2,Y1),G(X2,Y2)]);
            undis_img(i,j) = uint8((1/(X2-X1)/(Y2-Y1))*nni_val* ...
                        [X2*Y2,-Y2,-X2,1;
                         -X2*Y1,Y1,X2,-1;
                         -X1*Y2,Y2,X1,-1;
                         X1*Y1,-Y1,-X1,1]*[1;Xd;Yd;Xd*Yd]);
        end
    end
    imshow(undis_img)
    %% draw a cube
    Pw_list = [[Xw(4,4),Yw(4,4),0];
                [Xw(6,4),Yw(6,4),0];
                [Xw(6,6),Yw(6,6),0];
                [Xw(4,6),Yw(4,6),0];
                [Xw(4,4),Yw(4,4),-0.08];
                [Xw(6,4),Yw(6,4),-0.08];
                [Xw(6,6),Yw(6,6),-0.08];
                [Xw(4,6),Yw(4,6),-0.08];
                ];
    for i=1:8
        Pp(i,1:3) = projectPoints(T,K,Pw_list(i,:));
    end
    verts = Pp(1,1:2);
    for i = 2:8
        verts = [verts;Pp(i,1:2)];
    end

    % 6¸öÃæ¶¥µã±àºÅ
    faces=[1 2 3 4;
           1 2 6 5;
           2 3 7 6;
           3 4 8 7;
           4 1 5 8;
           5 6 7 8];

    % draw
    patch('Faces',faces,'Vertices',verts,'Facecolor','none','LineWidth',1.5,'EdgeColor','red')
    
    % video
    frame = getframe(gcf);
    im = frame2im(frame);
    filename = [sprintf('%03d',m) '.jpg'];
    fullname = fullfile(workingDir,'images',filename);
    imwrite(im,fullname) 
end
% 
imageNames = dir(fullfile(workingDir,'images','*.jpg'));
imageNames = {imageNames.name}';
outputVideo = VideoWriter(fullfile(workingDir,'shuttle_out.avi'));
outputVideo.FrameRate = shuttleVideo.FrameRate;
open(outputVideo);
for ii = 1:length(imageNames)
   img = imread(fullfile(workingDir,'images',imageNames{ii}));
   writeVideo(outputVideo,img)
end
close(outputVideo)
shuttleAvi = VideoReader(fullfile(workingDir,'shuttle_out.avi'));
ii = 1;
while hasFrame(shuttleAvi)
   mov(ii) = im2frame(readFrame(shuttleAvi));
   ii = ii+1;
end
movie(mov,1,shuttleAvi.FrameRate)