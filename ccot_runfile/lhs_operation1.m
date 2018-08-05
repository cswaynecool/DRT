function hf_out = lhs_operation1(hf, samplesf, reg_filter, sample_weights, feature_reg,binary_mask,patch_mask,frame,feature_input,cnna)

if length(hf)==5
     feature_input.set_hf_5(real(permute(hf{1},[2 1 3 4])), imag(permute(hf{1},[2 1 3 4])) , real(permute(hf{2},[2 1 3 4])), imag(permute(hf{2},[2 1 3 4])),...
      real(permute(hf{3},[2 1 3 4])), imag(permute(hf{3},[2 1 3 4])) , real(permute(hf{4},[2 1 3 4])), imag(permute(hf{4},[2 1 3 4])) ,real(permute(hf{5},[2 1 3 4])), imag(permute(hf{5},[2 1 3 4]))   ); 
else
     feature_input.set_hf_4(real(permute(hf{1},[2 1 3 4])), imag(permute(hf{1},[2 1 3 4])) , real(permute(hf{2},[2 1 3 4])), imag(permute(hf{2},[2 1 3 4])),...
      real(permute(hf{3},[2 1 3 4])), imag(permute(hf{3},[2 1 3 4])) , real(permute(hf{4},[2 1 3 4])), imag(permute(hf{4},[2 1 3 4])) ); 
 end

   cnna.net.forward_prefilled();


for block_id=1:length(hf)
      test1=cnna.net.params('conv5_f1',block_id).get_data();
      test1=permute(test1,[2 1 3 4]);
      hf_out{1,1,block_id}=test1(:,:,:,1)+1i*test1(:,:,:,2);
end


end