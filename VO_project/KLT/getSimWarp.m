function W = getSimWarp(dx, dy, alpha_deg, lambda)
    % alpha given in degrees, as indicated
    D2R = pi / 180;
    alpha = alpha_deg * D2R;
    W = [cos(alpha), -sin(alpha), dx;
            sin(alpha), cos(alpha), dy];
    W = lambda * W;
end
