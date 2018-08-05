xl1 = cellfun(@(x) reshape(x, [], size(x,3)), xl, 'uniformoutput', false);
x_mean=cellfun(@(x) permute(mean(x, 1),[1 3 2]), xl1, 'uniformoutput', false);

for block_id=1:length(xl)
    tmp=repmat(x_mean{block_id},[feature_sz(block_id,1)  feature_sz(block_id,2) 1]);
    x_mean{1,1,block_id}=tmp;
end

xl1 = cellfun(@(x) bsxfun(@minus, x, mean(x, 1)), xl1, 'uniformoutput', false);
[projection_matrix, ~, ~] = cellfun(@(x) svd(x' * x), xl1, 'uniformoutput', false);
for block_id=1:length(xl)
     compressed_dim_cell{1,1,block_id}=params.compressed_dim(block_id);
end
projection_matrix = cellfun(@(P, dim) single(P(:,1:dim)), projection_matrix, compressed_dim_cell, 'uniformoutput', false);