function I = warpImage(I_R, W)
    row = size(I_R,1);
    col = size(I_R,2);
    I = zeros(row,col);
    for i = 1:row
        for j = 1:col
            a = round(W(i,j,2));
            b = round(W(i,j,1));
            if a >= 1 && a <= row && b >=1 && b <= col
                I(i,j) = I_R(a,b);
            end    
        end
    end



end