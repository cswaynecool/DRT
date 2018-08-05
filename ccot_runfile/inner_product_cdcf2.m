function ip = inner_product_cdcf2(xf, yf)

% Computes the inner product between two filters.
for block_id=1:length(xf);
    tmp1=full_fourier_coeff(xf{1,1,block_id});
    tmp2=ifftshift(ifftshift(tmp1,1),2);
    tmp3=real(ifft2(tmp2));
    tmp4=full_fourier_coeff(yf{1,1,block_id});
    tmp5=ifftshift(ifftshift(tmp4,1),2);
    tmp6=real(ifft2(tmp5));
    ip_cell{1,1,block_id}=real(tmp3(:)'*tmp6(:))*size(tmp1,1)*size(tmp1,2);
end


% ip_cell = cellfun(@(xf, yf) real(2*(xf(:)' * yf(:)) - reshape(xf(:,end,:), [], 1, 1)' * reshape(yf(:,end,:), [], 1, 1)), xf, yf, 'uniformoutput', false');
ip = sum(cell2mat(ip_cell));

% ip_cell = cellfun(@(xf, yf) real(xf(:)' * yf(:)), xf, yf, 'uniformoutput', false');
% ip = sum(cell2mat(ip_cell));