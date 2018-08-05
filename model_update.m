rand('state',0);
if frame==1
       %% inialize the parameters for caffe toolkit
       index_tmp=1:filter_sz(3,1)*(filter_sz(3,2)+1)/2;
       index_tmp=reshape(index_tmp,[(filter_sz(3,1)+1)/2, filter_sz(3,2)]);
       index_tmp=permute(index_tmp,[2 1 3 4]);
       index_tmp=padarray(index_tmp,pad_sz{3});
       L_index=index_tmp(:,1:(filter_sz(1,1)+1)/2);
       L_index=permute(L_index,[2 1 3 4]);
       L_index1=find(L_index(:));
       L_index1=reshape(L_index1,[(filter_sz(3,1)+1)/2, filter_sz(3,2)]);
       feature_input.set_L_index(L_index,L_index1);
        num_training_samples=0;
       frames_since_last_train = inf;
        max_train_samples=params.nSamples;
        prior_weights = zeros(max_train_samples,1, 'single');
        minimum_sample_weight = params.learning_rate*(1-params.learning_rate)^(2*max_train_samples);
        score_matrix = inf(max_train_samples, 'single');
        latest_ind = [];
        samplesf = cell(1, 1, num_feature_blocks);
        sample_weights = prior_weights;
        val_index(1)=0; 
        val_index(2)=filter_sz(1,1)*(filter_sz(1,2)+1)/2*(params.frag_num+1)*params.compressed_dim(1);
        val_index(4)=val_index(2)+filter_sz(1,1)*(filter_sz(1,2)+1)/2*(params.frag_num+1)*(params.compressed_dim(2));
        if num_feature_blocks>4
            val_index(5)=val_index(4)+filter_sz(1,1)*(filter_sz(1,2)+1)/2*(params.frag_num+1)*(params.compressed_dim(4));
        end
        val_index(3)=0;
        
        val_index1(1)=0; 
        val_index1(2)=filter_sz(1,1)*filter_sz(1,2)*(params.frag_num+1)*params.compressed_dim(1);
        val_index1(4)=val_index1(2)+filter_sz(1,1)*filter_sz(1,2)*(params.frag_num+1)*(params.compressed_dim(2));
        if num_feature_blocks>4
            val_index1(5)=val_index1(4)+filter_sz(1,1)*filter_sz(1,2)*(params.frag_num+1)*(params.compressed_dim(4));
        end
        val_index1(3)=0;
        total_num=filter_sz(:,1).*filter_sz(:,2).*feature_dim;
        total_num1=filter_sz(:,1).*(filter_sz(:,2)+1)/2.*feature_dim;
        total_num(3,:)=0;
        total_num1(3,:)=0;
        total_num=sum(total_num);
        total_num1=sum(total_num1);
        total_num2=filter_sz(3,1).*filter_sz(3,2).*feature_dim(3);
        total_num3=filter_sz(3,1).*(filter_sz(3,2)+1)/2.*feature_dim(3);
        
        
        feature_input.set_index(val_index',val_index1',total_num,total_num1,total_num2,total_num3);
        for k=1:num_feature_blocks
               feature_input.inialize_blobs(k,params.nSamples,params.compressed_dim(k),filter_sz(k,1),filter_sz(k,2),params.frag_num);
              samplesf{k}=(single((complex(zeros(params.nSamples, params.compressed_dim(k),filter_sz(k,1),(filter_sz(k,2)+1)/2)))));
              samplesf{k}=single(samplesf{k});
        end
       feature_input.clear_memory_function(0);
        %% inialize the parameters for the fft operation in Caffe         
        fftshift_mask1=1:filter_sz(1,1)*( floor( filter_sz(1,2)/2)+1);
        fftshift_mask1=reshape(fftshift_mask1,[floor( filter_sz(1,2)/2)+1  filter_sz(1,1) ])-1;
        fftshift_mask1=permute(fftshift_mask1,[2 1]);
        fftshift_mask1=process_freq(fftshift_mask1,filter_sz(1,1),filter_sz(1,2));
        fftshift_mask1=fftshift(fftshift(fftshift_mask1,1),2);
        fftshift_mask1=fftshift_mask1(:, 1:floor(filter_sz(1,2)/2)+1 );  
        % 
        fftshift_mask2=1:filter_sz(3,1)*( floor( filter_sz(3,2)/2)+1);
        fftshift_mask2=reshape(fftshift_mask2,[floor( filter_sz(3,2)/2)+1  filter_sz(3,1) ])-1;
        fftshift_mask2=permute(fftshift_mask2,[2 1]);
        fftshift_mask2=process_freq(fftshift_mask2,filter_sz(3,1),filter_sz(3,2));
        fftshift_mask2=fftshift(fftshift(fftshift_mask2,1),2);
        fftshift_mask2=fftshift_mask2(:, 1:floor(filter_sz(3,2)/2)+1 );    
% 
%     
    ifftshift_mask1=1:filter_sz(1,1)*( floor( filter_sz(1,2)/2)+1);
    ifftshift_mask1=reshape(ifftshift_mask1,[floor( filter_sz(1,2)/2)+1  filter_sz(1,1) ] );
    ifftshift_mask1=permute(ifftshift_mask1,[2 1]);
    ifftshift_mask1= full_fourier_coeff_ifftshift(ifftshift_mask1);
    ifftshift_mask1=ifftshift(ifftshift(ifftshift_mask1,1),2);
    ifftshift_mask1=ifftshift_mask1(:, 1:floor(filter_sz(1,2)/2)+1 );      
%     
    ifftshift_mask2=1:filter_sz(3,1)*( floor( filter_sz(3,2)/2)+1); 
    ifftshift_mask2=reshape(ifftshift_mask2,[floor( filter_sz(3,2)/2)+1  filter_sz(3,1) ] );
    ifftshift_mask2=permute(ifftshift_mask2,[2 1]);
    ifftshift_mask2= full_fourier_coeff_ifftshift(ifftshift_mask2);
    ifftshift_mask2=ifftshift(ifftshift(ifftshift_mask2,1),2);
    ifftshift_mask2=ifftshift_mask2(:, 1:floor(filter_sz(3,2)/2)+1 );    
% 
    fftshift_mask1=permute(fftshift_mask1,[2 1]);
    fftshift_mask2=permute(fftshift_mask2,[2 1]);
    ifftshift_mask1=permute(ifftshift_mask1,[2 1]);
    ifftshift_mask2=permute(ifftshift_mask2,[2 1]);
    
    feature_input.set_reg_window(fftshift_mask1,fftshift_mask2,ifftshift_mask1,ifftshift_mask2, permute(binary_mask{1},[2 1 3 4]),permute(binary_mask{3},[2 1 3 4]));
    feature_input.set_patch_mask(  permute(patch_mask{1},[2 1 3 4]) , permute(patch_mask{3},[2 1 3 4])    );
    feature_input.set_binary_mask_adaptive(permute(binary_mask{1},[2 1 3 4]),permute(binary_mask{3},[2 1 3 4]));
end


for block_id=1:num_feature_blocks
       fft_input{block_id}=fft2(single((xl{block_id})));
end

xlf_proj_perm = cellfun(@(xf) permute(xf, [4 3 1 2]), xlf, 'uniformoutput', false);
dist_vector = find_cluster_distances(samplesf, xlf_proj_perm, num_feature_blocks, num_training_samples, max_train_samples, params);
 [merged_sample, new_cluster, merged_cluster_id, new_cluster_id, score_matrix, prior_weights,num_training_samples] = ...
            merge_clusters(samplesf, xlf_proj_perm, dist_vector, score_matrix, prior_weights,...
                           num_training_samples,num_feature_blocks,max_train_samples,minimum_sample_weight,params);
sample_weights = prior_weights;
  feature_input.input_sample_weight(sample_weights);
for k = 1:num_feature_blocks

       [test2,test3]=feature_input.get_samplesf(1);
      if merged_cluster_id > 0
                samplesf{k}(merged_cluster_id,:,:,:) = merged_sample{k};
                feature_input.update_samplesf(real(permute(merged_sample{k},[4 3 2 1])),imag(permute(merged_sample{k},[4 3 2 1])),k,merged_cluster_id);
            end
            
            if new_cluster_id > 0
                samplesf{k}(new_cluster_id,:,:,:) = new_cluster{k};
                feature_input.update_samplesf(real(permute(new_cluster{k},[4 3 2 1])),imag(permute(new_cluster{k},[4 3 2 1])),k,new_cluster_id);
            end

end

if frame==1
         for block_id=1:num_feature_blocks
            monument{block_id}=(single(complex(zeros(filter_sz(block_id,1),filter_sz(block_id,2), feature_dim(block_id) ))));
            hf{1,1,block_id}=single(complex(zeros(filter_sz(block_id,1),(filter_sz(block_id,2)+1)/2,params.compressed_dim(block_id))));
             
         end
         loop_max=10000;
else
         loop_max=10;
end


train_tracker = (frame < params.skip_after_frame) || (frames_since_last_train >= params.train_gap);
if (train_tracker||frame==1)
                matlab_validation1;
                 frames_since_last_train = 0;
                 
else
     frames_since_last_train = frames_since_last_train+1;
end

      
      
      
      
      
      
      
      

                    




