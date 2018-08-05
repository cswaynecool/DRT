function ip = inner_product_cdcf1(xf, yf,cnn_third,feature_input)

% Computes the inner product between two filters.
if prod(size(xf))==4
    feature_input.set_xf_yf4(permute(real(xf{1}),[2 1 3 4]), permute(imag(xf{1}),[2 1 3 4]) , permute(real(xf{2}),[2 1 3 4]), permute(imag(xf{2}),[2 1 3 4]) ,...
    permute(real(xf{3}),[2 1 3 4]), permute(imag(xf{3}),[2 1 3 4]) ,permute(real(xf{4}),[2 1 3 4]), permute(imag(xf{4}),[2 1 3 4]),...
    permute(real(yf{1}),[2 1 3 4]), permute(imag(yf{1}),[2 1 3 4]) , permute(real(yf{2}),[2 1 3 4]), permute(imag(yf{2}),[2 1 3 4]) ,...
    permute(real(yf{3}),[2 1 3 4]), permute(imag(yf{3}),[2 1 3 4]) ,permute(real(yf{4}),[2 1 3 4]), permute(imag(yf{4}),[2 1 3 4]) );
else
    feature_input.set_xf_yf5(permute(real(xf{1}),[2 1 3 4]), permute(imag(xf{1}),[2 1 3 4]) , permute(real(xf{2}),[2 1 3 4]), permute(imag(xf{2}),[2 1 3 4]) ,...
    permute(real(xf{3}),[2 1 3 4]), permute(imag(xf{3}),[2 1 3 4]) ,permute(real(xf{4}),[2 1 3 4]), permute(imag(xf{4}),[2 1 3 4]) ,permute(real(xf{5}),[2 1 3 4]), permute(imag(xf{5}),[2 1 3 4]),...
    permute(real(yf{1}),[2 1 3 4]), permute(imag(yf{1}),[2 1 3 4]) , permute(real(yf{2}),[2 1 3 4]), permute(imag(yf{2}),[2 1 3 4]) ,...
    permute(real(yf{3}),[2 1 3 4]), permute(imag(yf{3}),[2 1 3 4]) ,permute(real(yf{4}),[2 1 3 4]), permute(imag(yf{4}),[2 1 3 4]) ,permute(real(yf{5}),[2 1 3 4]), permute(imag(yf{5}),[2 1 3 4]));
end
cnn_third.net.forward_prefilled();
output=cnn_third.net.blobs('conv5_f1');
output_data=output.get_data();

% for block_id=1:length(xf);
%     tmp1=full_fourier_coeff(xf{1,1,block_id});
%     tmp2=ifftshift(ifftshift(tmp1,1),2);
%     tmp3=real(ifft2(tmp2));
%     tmp4=full_fourier_coeff(yf{1,1,block_id});
%     tmp5=ifftshift(ifftshift(tmp4,1),2);
%     tmp6=real(ifft2(tmp5));
%     ip_cell{1,1,block_id}=gather(real(tmp3(:)'*tmp6(:)));
%     ip_cell{1,1,block_id}=gather(real(tmp3(:)'*tmp6(:))*size(tmp1,1)*size(tmp1,2));
% end


% ip_cell = cellfun(@(xf, yf) real(2*(xf(:)' * yf(:)) - reshape(xf(:,end,:), [], 1, 1)' * reshape(yf(:,end,:), [], 1, 1)), xf, yf, 'uniformoutput', false');
% ip = sum(cell2mat(ip_cell));
ip=output_data;
% ip_cell = cellfun(@(xf, yf) real(xf(:)' * yf(:)), xf, yf, 'uniformoutput', false');
% ip = sum(cell2mat(ip_cell));