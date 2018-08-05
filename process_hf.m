

cnnc.net.forward_prefilled();
output=cnnc.net.blobs('conv5_f1');
output_data=output.get_data();
output_data=permute(output_data,[2 1 3 4]);

for block_id=1:num_feature_blocks
             
             test1=cnnc.net.params('conv5_f1',block_id).get_data();
             test1=permute(test1,[2 1 3 4]);
             hf1{1,1,block_id}=test1(:,:,:,1)+1i*test1(:,:,:,2);   
             hf1{block_id}= symmetrize_filter1(hf1{block_id});
             hf1{block_id}=full_fourier_coeff(hf1{block_id});
end
clear H;
weight_f= hf1;