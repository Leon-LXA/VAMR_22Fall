function T=poseVectorToTransformationMatrix(pos)
    w = pos(1:3);
    theta = norm(w);
    k = w/theta;
    t = pos(4:6);
    R = eye(3) + sin(theta) * cp_mat(k) + (1-cos(theta)) * cp_mat(k)^2;
    T = [R,t'];
%     T = [T;[0,0,0,1]];
end