clear all
close all
clc

num_scales = 3; % Scales per octave.
num_octaves = 5; % Number of octaves.
sigma = 1.6;
contrast_threshold = 0.04;
image_file_1 = 'images/img_1.jpg';
image_file_2 = 'images/img_2.jpg';
rescale_factor = 0.2; % Rescaling of the original image for speed.

images = {getImage(image_file_1, rescale_factor),...
    getImage(image_file_2, rescale_factor)};

kpt_locations = cell(1, 2);
descriptors = cell(1, 2);

k = pow2(1.0 / num_scales);
for img_idx = 1:2
    % Write code to compute:
    % 1)    image pyramid. Number of images in the pyarmid equals
    %       'num_octaves'.
    img = images{1,img_idx};
    img_2 = img(1:2:end,1:2:end);
    img_3 = img_2(1:2:end,1:2:end);
    img_4 = img_3(1:2:end,1:2:end);
    img_5 = img_4(1:2:end,1:2:end);
%     imshow(img)
%     figure(2)
%     imshow(img_3)
    octave_1 = {img, img, img, img, img, img};
    octave_2 = {img_2, img_2, img_2, img_2, img_2, img_2};
    octave_3 = {img_3, img_3, img_3, img_3, img_3, img_3};
    octave_4 = {img_4, img_4, img_4, img_4, img_4, img_4};
    octave_5 = {img_5, img_5, img_5, img_5, img_5, img_5};
    octave = {octave_1,octave_2,octave_3,octave_4,octave_5};
    % 2)    blurred images for each octave. Each octave contains
    %       'num_scales + 3' blurred images.
%     test = octave{1,1}{1,1};
    
    for i = 1:num_octaves
        for j = 1: (num_scales+3)
            octave{1,i}{1,j} = imgaussfilt(octave{1,i}{1,j}, sigma * k^(j-2));
        end
    end

%     figure(2)
%     imshow(octave{1,1}{1,1}-test)
    
    % 3)    'num_scales + 2' difference of Gaussians for each octave.
    for i = 1:num_octaves
        for j = 1: (num_scales+2)
            DoG_octave{1,i}{1,j} = (octave{1,i}{1,j+1} - octave{1,i}{1,j});
%             for m = 1:size(DoG_octave{1,i}{1,j},1)
%                 for n = 1:size(DoG_octave{1,i}{1,j},2)
%                     if DoG_octave{1,i}{1,j}(m,n) < contrast_threshold
%                         DoG_octave{1,i}{1,j}(m,n) = 0;
%                     end
%                 end
%             end
        end
    end

    
    figure(img_idx)
%     imshow(DoG_octave{1,1}{1,1})
    imshow(img)
    hold on;
    
    % 4)    Compute the keypoints with non-maximum suppression and
    %       discard candidates with the contrast threshold.
 
    
    for i = 1:num_octaves
        for j = 1: num_scales
            [gmag,Gdir] = imgradient(octave{1,i}{1,j+1});
            
            for m = 8:size(DoG_octave{1,i}{1,j},1)-8
                for n = 8:size(DoG_octave{1,i}{1,j},2)-8
                    center = DoG_octave{1,i}{1,j+1}(m,n);
                    last_center = DoG_octave{1,i}{1,j}(m,n);
                    next_center = DoG_octave{1,i}{1,j+2}(m,n);
                    
                    center_layer = DoG_octave{1,i}{1,j+1}(m-1:m+1,n-1:n+1);
                    last_layer = DoG_octave{1,i}{1,j}(m-1:m+1,n-1:n+1);
                    next_layer = DoG_octave{1,i}{1,j+2}(m-1:m+1,n-1:n+1);
                    max_center_layer = max(max(center_layer));
                    max_last_layer = max(max(last_layer));
                    max_next_layer = max(max(next_layer));
                    if(center == max([max_center_layer,max_last_layer,max_next_layer]) && center > contrast_threshold ...
                       || last_center == max([max_center_layer,max_last_layer]) && last_center > contrast_threshold ...
                       || next_center == max([max_center_layer,max_next_layer]) && next_center > contrast_threshold)
                        x_idx = 2^(i-1)*m;
                        y_idx = 2^(i-1)*n;
                        kpt_locations{1,img_idx} = [kpt_locations{1,img_idx};[x_idx,y_idx,i]]; % index of the corresponding scale
                        scatter(y_idx,x_idx);
                        hold on;
                        
%                         descriptors{1,img_idx} = grad_img(m-7:m+8,n-7:n+8);
                        extracted_mag = gmag(m-7:m+8,n-7:n+8);
                        extracted_dir = Gdir(m-7:m+8,n-7:n+8);
                        normed_mag = imgaussfilt(extracted_mag, 1.5*16);
                        hist = [];
                        for i_extra = linspace(1,13,4)
                            for j_extra = linspace(1,13,4)
                                mat_mag = normed_mag(i_extra:i_extra+3,j_extra:j_extra+3);
                                vec_mag = mat_mag(:);
                                mat_dir = extracted_dir(i_extra:i_extra+3,j_extra:j_extra+3);
                                vec_dir = mat_dir(:);
                                hist = [hist,weightedhistc(vec_dir,vec_mag,linspace(-180,135,8))];
                            end
                        end
                        hist = hist/norm(hist);
                        descriptors{1,img_idx} = [descriptors{1,img_idx};hist];
                    end
                end
            end
        end
    end
    % 5)    Given the blurred images and keypoints, compute the
    %       descriptors. Discard keypoints/descriptors that are too close
    %       to the boundary of the image. Hence, you will most likely
    %       lose some keypoints that you have computed earlier.
    
    
end

% Finally, match the descriptors using the function 'matchFeatures' and
% visualize the matches with the function 'showMatchedFeatures'.
% If you want, you can also implement the matching procedure yourself using
% 'knnsearch'.
indexPairs = matchFeatures(descriptors{1,1},descriptors{1,2},'MatchThreshold',100,'MaxRatio',0.5);
matchedPoints1 = [kpt_locations{1,1}(indexPairs(:,1),2),...
                    kpt_locations{1,1}(indexPairs(:,1),1)]; 
matchedPoints2 = [kpt_locations{1,2}(indexPairs(:,2),2),...
                    kpt_locations{1,2}(indexPairs(:,2),1)]; 

figure; 
showMatchedFeatures(images{1,1},images{1,2},matchedPoints1,matchedPoints2,'montage');
