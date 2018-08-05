new_sample_energy = cellfun(@(xlf) abs(xlf .* conj(xlf)), xlf, 'uniformoutput', false); 
reg_energy{1,1,1}=0.1; reg_energy{1,1,2}=0.1; reg_energy{1,1,3}=0.1; reg_energy{1,1,4}=0.1; %reg_energy{1,1,5}=0.1;

feature_input.set_frame_id(frame);
if frame == 1 
        % Initialize Conjugate Gradient parameters
        p = [];
        rho = [];
     
        max_CG_iter = params.init_max_CG_iter;
        sample_energy = new_sample_energy;
 else
        max_CG_iter = params.max_CG_iter;
        
        if params.CG_forgetting_rate == inf || params.learning_rate >= 1
            max_CG_iter=params.init_max_CG_iter;
        else
            rho = rho / (1-params.learning_rate)^params.CG_forgetting_rate;
        end
        
        % Update the approximate average sample energy using the learning
        % rate. This is only used to construct the preconditioner.
        sample_energy = cellfun(@(se, nse) (1 - params.learning_rate) * se + params.learning_rate * nse, sample_energy, new_sample_energy, 'uniformoutput', false);
 end

samplesf_tmp=samplesf;

 rhs_samplef = cellfun(@(xf) permute(mtimesx(sample_weights, 'T', xf, 'speed'), [3 4 2 1]), samplesf_tmp, 'uniformoutput', false);
 rhs_samplef = cellfun(@(xf, yf) bsxfun(@times, (xf), yf), rhs_samplef, yf, 'uniformoutput', false);
 rhs_samplef= full_fourier_coeff(rhs_samplef);
 right_hand_tmp=rhs_samplef;


if frame==1
    out_loop=10;
    inner_loop=5;
    lb = 0.5*ones(9,1);
    ub =1.5*ones(9,1);
    reliability_val=ones(params.frag_num,1);
      
else
    out_loop=1;
    inner_loop=5;
end

for ite_out=1:out_loop
    
    compute_right_hand;
    

     [hf, flag, relres, iter, res_norms, p, rho] = pcg_ccot(...
        @(x) lhs_operation1(x, samplesf_tmp, reg_filter, sample_weights, feature_reg,binary_mask,patch_mask,frame,feature_input,cnna),...
        rhs_samplef, CG_tol, max_CG_iter, ...
        [], ...
         [], hf, p, rho,cnn_third,feature_input);

 for block_id=1:length(hf)
    hf{block_id}=gather(single(hf{block_id}));
    samplesf_tmp{block_id}=gather(single(samplesf_tmp{block_id}));
    hf{block_id}=gather(hf{block_id});
 end  
 hf_tmp=symmetrize_filter(hf);
 hf_tmp=full_fourier_coeff(hf_tmp);
  if length(hf)==5
  feature_input.set_H_5(permute(real(ifft2(   ifftshift( ifftshift(hf_tmp{1},1),2)   )),[2 1 3 4]),   permute(real(ifft2(   ifftshift( ifftshift(hf_tmp{2},1),2)   )),[2 1 3 4]), permute(real(ifft2(   ifftshift( ifftshift(hf_tmp{3},1),2)   )),[2 1 3 4]) ,...
        permute(real(ifft2(   ifftshift( ifftshift(hf_tmp{4},1),2)   )),[2 1 3 4]), permute(real(ifft2(   ifftshift( ifftshift(hf_tmp{5},1),2)   )),[2 1 3 4]),params.frag_num);
  else
      feature_input.set_H_4(permute(real(ifft2(   ifftshift( ifftshift(hf_tmp{1},1),2)   )),[2 1 3 4]),   permute(real(ifft2(   ifftshift( ifftshift(hf_tmp{2},1),2)   )),[2 1 3 4]), permute(real(ifft2(   ifftshift( ifftshift(hf_tmp{3},1),2)   )),[2 1 3 4]) ,...
        permute(real(ifft2(   ifftshift( ifftshift(hf_tmp{4},1),2)   )),[2 1 3 4]),params.frag_num);
  end
        cnnb.net.forward_prefilled();
        output=cnnb.net.blobs('conv5_f1');
        output_data=output.get_data();
        output_data=permute(output_data,[2 1 3 4]);
    
        for frag_id=1:params.frag_num
            tmp3{frag_id}=output_data(:,:,:,frag_id);
            tmp4{frag_id}=output_data(:,:,:,frag_id+params.frag_num);
        end
    
        
        for frag_id=1:params.frag_num
              tmp_tmp=bsxfun(@times,tmp4{frag_id},y_real{1});
              right_hand(frag_id,1)=sum(tmp_tmp(:));
        end
        
         for frag_id1=1:params.frag_num
             for frag_id2=1:params.frag_num
                   tmp=tmp3{frag_id1}.*tmp3{frag_id2};
                   A(frag_id1,frag_id2)=sum(tmp(:));
             end
         end
 

     if frame>1
     
         
         A_total=A;
         reliability_val_old=reliability_val;
            A1 = [];
            b = [];
            Aeq = [];
            beq = [];
            
       
            if sum(reliability_val)>2.99*9
                  lb = 2*ones(9,1);
                 ub =6*ones(9,1);
            elseif sum(reliability_val)>1.49*9
                 lb = 1*ones(9,1);
                 ub =3*ones(9,1);
            end
            
options = optimoptions('lsqlin','Algorithm','trust-region-reflective');
%            [x,resnorm,residual,exitflag,output,lambda] = lsqlin(double(A),double(right_hand),A1,b,Aeq,beq,lb,ub,double(reliability_val),options);
% opts = optimoptions('quadprog','Algorithm','trust-region-reflective','Display','off');
opts = optimoptions('quadprog','Algorithm','active-set','Display','off');

             x = quadprog(double(A),double(-1*right_hand),[],[],[],[],lb,ub,double(reliability_val),opts);
               reliability_val=x;
     end

          index=reliability_val<0;
           reliability_val(index)=0;
          for block_id=1:num_feature_blocks
              binary_mask{block_id}=binary_mask{block_id}*0;
              for frag_id=1:params.frag_num
                 binary_mask{block_id}=binary_mask{block_id}+patch_mask{block_id}(:,:,frag_id)*reliability_val(frag_id);
              end
          end
         feature_input.set_binary_mask_adaptive(permute(binary_mask{1},[2 1 3 4]),permute(binary_mask{3},[2 1 3 4]));
end
              

hf1= symmetrize_filter(hf);

if length(hf)==5
     feature_input.set_hf_5(real(permute(hf1{1},[2 1 3 4])), imag(permute(hf1{1},[2 1 3 4])) , real(permute(hf1{2},[2 1 3 4])), imag(permute(hf1{2},[2 1 3 4])),...
      real(permute(hf1{3},[2 1 3 4])), imag(permute(hf1{3},[2 1 3 4])) , real(permute(hf1{4},[2 1 3 4])), imag(permute(hf1{4},[2 1 3 4])) ,real(permute(hf1{5},[2 1 3 4])), imag(permute(hf1{5},[2 1 3 4]))   ); 
else
     feature_input.set_hf_4(real(permute(hf1{1},[2 1 3 4])), imag(permute(hf1{1},[2 1 3 4])) , real(permute(hf1{2},[2 1 3 4])), imag(permute(hf1{2},[2 1 3 4])),...
      real(permute(hf1{3},[2 1 3 4])), imag(permute(hf1{3},[2 1 3 4])) , real(permute(hf1{4},[2 1 3 4])), imag(permute(hf1{4},[2 1 3 4])) ); 
 end


hf1=full_fourier_coeff(hf1);
process_hf;