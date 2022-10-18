function M = estimatePoseDLT(p, P, K)
    for i = 1:12
        p_temp = K \ [p(2*i-1);p(2*i);1];
        p(2*i-1:2*i) = p_temp(1:2)/p_temp(3);
    end
%     P
    % implement Q
    Q = zeros(24,12);
    for i = 1:12
        Q(2*i-1:2*i,:) = [P(i,1), P(i,2), P(i,3), 1, 0, 0, 0, 0, -p(2*i-1)*P(i,1), -p(2*i-1)*P(i,2), -p(2*i-1)*P(i,3), -p(2*i-1);
                          0, 0, 0, 0, P(i,1), P(i,2) ,P(i,3), 1, -p(2*i)*P(i,1), -p(2*i)*P(i,2), -p(2*i)*P(i,3), -p(2*i)];
    end
%     Q
    
    [U,S,V] = svd(Q);
    M = V(:,12);
    
    M = reshape(M,[4,3]);
    M = M';
    if(M(3,4) < 0)
        M = -M;
    end
%     M = M/ det(M(:,1:3))^(1/3);
    M = M/norm(M(:,1:3));

end