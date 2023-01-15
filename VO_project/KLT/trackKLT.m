function [p_t] = trackKLT(I_R, I, x_T, r_T, num_iters)
% I_R: reference image, I: image to track point in
% x_T: point to track, expressed as [x y]=[col row], r_T: radius of patch to track
% Return: W(2x3) - final W estimate

    len_patch = 2*r_T + 1;
    eps = 5e-4;
    p_t = getSimWarp(0,0,0,1);
    patch_ref = getWarpedPatch(I_R, p_t, x_T, r_T);
    patch_ref = patch_ref(:);

    for iter = 1:num_iters
        patch_margin = getWarpedPatch(I, p_t, x_T, r_T + 1);
        % get Grad of warped image, each size n*n
        grad_Ix = conv2(1, [1 0 -1], patch_margin(2:end-1, :), 'valid');
        grad_Iy = conv2([1 0 -1], 1, patch_margin(:, 2:end-1), 'valid');
        
        % calculate H = r_I / r_p = grad_I * r_W/r_p
        rI_rp = zeros(len_patch*len_patch, 6);
        for u = 1:len_patch
            for v = 1:len_patch
                idx = (u-1)*len_patch+v;
                % round I / round p = (round I / round W) * (round W / round p)
                rW_rp = kron([u - (r_T+1), v - (r_T+1), 1], eye(2));
                % important!!! the order of x and y!!
                rI_rp(idx, :) = [grad_Ix(v, u), grad_Iy(v, u)] * rW_rp;
            end 
        end
        
        Hessian = rI_rp' * rI_rp;
        % get difference between two imgs
        patch_t = patch_margin(2:end-1, 2:end-1);
        % 6*1 = 6*6 x 6*n^2 x n^2*1
        delta_p = (Hessian \ rI_rp') * (patch_ref - patch_t(:));
        warning('off', 'MATLAB:singularMatrix');
        % update p
        p_t = p_t + reshape(delta_p, [2, 3]);
        if(norm(delta_p) < eps)
            return
        end
        
    end
end
