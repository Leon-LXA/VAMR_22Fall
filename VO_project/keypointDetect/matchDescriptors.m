function matches = matchDescriptors(query_descriptors, database_descriptors, lambda)
% The descriptor vectors are MxQ and MxD where M is the descriptor dimension 
% and Q and D the amount of query and database descriptors respectively. 

% Returns
% matches [1*Q]:the i-th coefficient is the index of the
% db-descriptor which matches to the i-th query descriptor.
% 0 if there is no db-descriptor with an SSD < lambda * min(SSD).
% No two non-zero elements of matches will be equal.

    % Q = size(query_descriptors, 2); 
    % D = size(database_descriptors, 2);
    [dists, matches] = pdist2(...
        double(database_descriptors)', double(query_descriptors)',...
        'euclidean', 'Smallest', 1);

    sorted_dists = sort(dists);
    sorted_dists = sorted_dists(sorted_dists~=0);
    min_non_zero_dist = sorted_dists(1);
    matches(dists >= lambda * min_non_zero_dist) = 0;
    
    unique_matches = zeros(size(matches));
    [~,unique_match_idxs,~] = unique(matches, 'stable');
    unique_matches(unique_match_idxs) = matches(unique_match_idxs);
    matches = unique_matches;
end
