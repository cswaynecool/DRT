function x = project_sample(x, P,x_mean)

if ~isempty(P)
    %% 首先减去均值
%     x = cellfun(@(x) bsxfun(@minus, x, mean(x, 1)), x, 'uniformoutput', false);
     x = cellfun(@(x, x1) bsxfun(@minus, x, x1), x, x_mean, 'uniformoutput', false);
    %% 然后乘以降维矩阵
    x = cellfun(@(x, P) permute(mtimesx(permute(x, [4 3 1 2]), P, 'speed'), [3 4 2 1]), x, P, 'uniformoutput', false);
end