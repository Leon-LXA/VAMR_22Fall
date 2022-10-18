clc, clear;

%% videw setup
workingDir = tempname;
mkdir(workingDir);
mkdir(workingDir,'images');
shuttleVideo = VideoReader('shuttle.avi');

%% Read Image and matrix
A = imread('./data/images_undistorted/img_0001.jpg');
% imshow(A);
G = rgb2gray(A);
figure(1);
imshow(G);
hold on;

K = [420.506712, 0,          355.208298;
    0,         420.610940,  250.336787;
    0,         0,          1.];

pose_list = load('./data/poses.txt');
% pose = [-0.372483192214,0.0397022486165,0.0650393402332,-0.107035863625,-0.147065242923,0.398512498053];

%% corner pts pos
xw = 0:0.04:0.32;
yw = 0:0.04:0.20;
[Xw, Yw] = meshgrid(xw,yw);

%% get Pp
T = poseVectorToTransformationMatrix(pose_list(1,:))
for i=1:54
    Pp = projectPoints(T,K,[Xw(i),Yw(i),0]);
    scatter(Pp(1),Pp(2),'filled','r')
    hold on;
end

%% draw a cube
Pw_list = [[Xw(1,1),Yw(1,1),0];
            [Xw(3,1),Yw(3,1),0];
            [Xw(3,3),Yw(3,3),0];
            [Xw(1,3),Yw(1,3),0];
            [Xw(1,1),Yw(1,1),-0.08];
            [Xw(3,1),Yw(3,1),-0.08];
            [Xw(3,3),Yw(3,3),-0.08];
            [Xw(1,3),Yw(1,3),-0.08];
            ];
for i=1:8
    Pp(i,1:3) = projectPoints(T,K,Pw_list(i,:));
end
verts = Pp(1,1:2);
for i = 2:8
    verts = [verts;Pp(i,1:2)];
end

% 6个面顶点编号
faces=[1 2 3 4;
       1 2 6 5;
       2 3 7 6;
       3 4 8 7;
       4 1 5 8;
       5 6 7 8];

% draw
patch('Faces',faces,'Vertices',verts,'Facecolor','none','LineWidth',2.5,'EdgeColor','red')

% 
% %% video
% for j = 1:30
%     close all;
%     A = imread('./data/images_undistorted/img_0001.jpg');
%     G = rgb2gray(A);
%     imshow(G);
%     hold on;
%     %% get Pp
%     T = poseVectorToTransformationMatrix(pose_list(j,:));
%     for i=1:54
%         Pp = projectPoints(T,K,[Xw(i),Yw(i),0]);
%         scatter(Pp(1),Pp(2),'filled','r')
%         hold on;
%     end
% 
%     %% draw a cube
%     Pw_list = [[Xw(1,1),Yw(1,1),0];
%             [Xw(3,1),Yw(3,1),0];
%             [Xw(3,3),Yw(3,3),0];
%             [Xw(1,3),Yw(1,3),0];
%             [Xw(1,1),Yw(1,1),-0.08];
%             [Xw(3,1),Yw(3,1),-0.08];
%             [Xw(3,3),Yw(3,3),-0.08];
%             [Xw(1,3),Yw(1,3),-0.08];
%             ];
%     for i=1:8
%         Pp(i,1:3) = projectPoints(T,K,Pw_list(i,:));
%     end
%     verts = Pp(1,1:2);
%     for i = 2:8
%         verts = [verts;Pp(i,1:2)];
%     end
% 
%     % 6个面顶点编号
%     faces=[1 2 3 4;
%            1 2 6 5;
%            2 3 7 6;
%            3 4 8 7;
%            4 1 5 8;
%            5 6 7 8];
% 
%     % draw
%     patch('Faces',faces,'Vertices',verts,'Facecolor','none','LineWidth',1.5,'EdgeColor','red')
%     
%     frame = getframe(gcf);
%     im = frame2im(frame);
%     filename = [sprintf('%03d',j) '.jpg'];
%     fullname = fullfile(workingDir,'images',filename);
%     imwrite(im,fullname) 
% 
%     
%     
% end
% 
% imageNames = dir(fullfile(workingDir,'images','*.jpg'));
% imageNames = {imageNames.name}';
% outputVideo = VideoWriter(fullfile(workingDir,'shuttle_out.avi'));
% outputVideo.FrameRate = shuttleVideo.FrameRate;
% open(outputVideo);
% for ii = 1:length(imageNames)
%    img = imread(fullfile(workingDir,'images',imageNames{ii}));
%    writeVideo(outputVideo,img)
% end
% close(outputVideo)
% shuttleAvi = VideoReader(fullfile(workingDir,'shuttle_out.avi'));
% ii = 1;
% while hasFrame(shuttleAvi)
%    mov(ii) = im2frame(readFrame(shuttleAvi));
%    ii = ii+1;
% end
% movie(mov,1,shuttleAvi.FrameRate)