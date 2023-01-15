function patch = getWarpedPatch(I, W, x_T, r_T)
% x_T is 1x2 and contains [x_T y_T] as defined in the statement. patch is
% (2*r_T+1)x(2*r_T+1) and arranged consistently with the input image I.
    max_coords = size(I);
    patch = zeros(2*r_T+1); bias = r_T+1;
    x_T = x_T';
    interpolation = true;

    for u = -r_T  : r_T
        for v = -r_T  : r_T
            warp_coord = x_T + W * [v; u; 1];
            w_u = floor(warp_coord(2)); % row idx
            w_v = floor(warp_coord(1)); % col idx
            if(all([w_u, w_v] > [1 1] & [w_u, w_v] < max_coords))
                if(interpolation == true)
                    weight = warp_coord - [w_v; w_u];
                    nni_val = [I(w_u, w_v), I(w_u, w_v+1);
                                    I(w_u+1, w_v), I(w_u+1, w_v+1)];
                    bi_val = [1-weight(2), weight(2)] * double(nni_val)/255 * [1-weight(1); weight(1)];
                    patch(u+bias, v+bias) = bi_val;
                else
                    patch(u+bias, v+bias) = I(w_u, w_v);
                end
            end
        end
    end
end