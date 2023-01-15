function [best_guess_history, max_num_inliers_history] = ...
    parabolaRansac(data, max_noise)
% data is 2xN with the data points given column-wise, 
% best_guess_history is 3xnum_iterations with the polynome coefficients 
%   from polyfit of the BEST GUESS SO FAR at each iteration columnwise and
% max_num_inliers_history is 1xnum_iterations, with the inlier count of the
%   BEST GUESS SO FAR at each iteration.
rng(2);
N = size(data,2);
iter_num = 100;
max_num_inliers = 0;
max_num_inliers_history = zeros(1,iter_num);
best_guess_history = zeros(3,iter_num);

for i = 1:iter_num
    sample_pt = datasample(data',3,'Replace',false); % 3*2
% X = [sample_pt(1,1)^2, sample_pt(1,1), 1;
%      sample_pt(2,1)^2, sample_pt(2,1), 1;
%      sample_pt(3,1)^2, sample_pt(3,1), 1];
% Y = [sample_pt(1,2);sample_pt(2,2);sample_pt(3,2)];
% para = (X'*X)\A'*Y;
    x = sample_pt(:,1);
    y = sample_pt(:,2);
    p = polyfit(x,y,2);

% compute error
    x_all = data(1,:);
    y_all = data(2,:);

    y_predict = polyval(p,x_all);
    y_delta = abs(y_predict - y_all);

    k = find(y_delta < max_noise);
    num_inliers = size(k,2);
    if num_inliers > max_num_inliers
        max_num_inliers = num_inliers;
        max_num_inliers_history(i) = max_num_inliers;
        best_guess_history(:,i) = p;
    else
        max_num_inliers_history(i) = max_num_inliers_history(i-1);
        best_guess_history(:,i) = best_guess_history(:,i-1);
    end
end


end
