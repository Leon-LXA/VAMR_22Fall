function W = getSimWarp(dx, dy, alpha_deg, lambda)
% alpha given in degrees, as indicated
alpha = alpha_deg * pi / 180;
mat = [cos(alpha) -sin(alpha);
    sin(alpha) cos(alpha)];

% for x = 1:1500
%     for y = 1:500
%         W(x,y,:) = lambda * (mat*[x;y]+[dx;dy]);
%     end
% end
x = 1:1500;
y = 1:500;
[X,Y] = meshgrid(x,y);
W(:,:,1) = lambda * (cos(alpha).*X - sin(alpha).*Y + dx);
W(:,:,2) = lambda * (sin(alpha).*X + cos(alpha).*Y + dy);



end