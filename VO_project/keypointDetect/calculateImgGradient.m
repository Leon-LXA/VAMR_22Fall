% calculate corner respond
function [sum_IxIx, sum_IyIy, sum_IxIy] = calculateImgGradient(img, patch_size)
    %return - gradient conv with box/gaussian filter
    
    % calculate the gradient of the img using sobel filter
    Sobel_x = [-1, 0, 1; -2 0 2; -1 0 1];
    Sobel_y = [-1, -2 , -1; 0, 0, 0; 1, 2, 1];
    Ix = conv2(img, Sobel_x, 'valid'); %(h-2, w-2)
    Iy = conv2(img, Sobel_y, 'valid'); %(h-2, w-2)
    IxIx = Ix .^2;
    IyIy = Iy .^2;
    IxIy = Ix .* Iy;
    box = ones(patch_size, patch_size);
     %(h - patch_r, w - patch_r)
    sum_IxIx = conv2(IxIx, box, 'valid');
    sum_IyIy = conv2(IyIy, box, 'valid');
    sum_IxIy = conv2(IxIy, box, 'valid');
end