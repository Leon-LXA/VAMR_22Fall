function scores = harris(img, patch_size, kappa)
    sobel_x = [-1 0 1;
               -2 0 2;
               -1 0 1];
    sobel_y = [-1 -2 -1;
                0  0  0;
                1  2  1];
    Ix = conv2(img, sobel_x, 'same');
    Iy = conv2(img, sobel_y, 'same');
    M_11 = Ix.^2;
    M_12 = Ix.* Iy;
    M_21 = M_12;
    M_22 = Iy.^2;
    
    scores = (M_11.*M_22 - M_12.* M_21) - kappa * (M_11+M_22).^2;
    scores(scores<0) = 0;
end