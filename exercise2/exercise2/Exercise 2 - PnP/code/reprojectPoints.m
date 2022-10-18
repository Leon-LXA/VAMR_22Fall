function p_reprojected = reprojectPoints(P, M, K)
    p_reprojected = zeros(12,2);
    for i = 1:size(P,1)
        p_tilde = K * M * [P(i,1); P(i,2); P(i,3); 1];
%         M * [P(i,1); P(i,2); P(i,3); 1]
        p_reprojected(i,:) = p_tilde(1:2)/p_tilde(3);
    end
end