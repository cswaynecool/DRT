function results = DRT(seq)

cleanupObj = onCleanup(@cleanupFun);
rand('state', 0);
a=1;
set_tracker_param;
params=testing();
net = load(['networks/imagenet-vgg-m-2048.mat']);
net = vl_simplenn_move(net, 'gpu');
%% read images
num_z = 4;
im = imread([seq.path,seq.s_frames(1).name]);

[height,width]=size(im(:,:,1));
CG_tol = params.CG_tol;

params.wsize = [seq.init_rect(1,4), seq.init_rect(1,3)];
params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2);
pos = floor(params.init_pos(:)');
target_sz = floor(params.wsize(:)');
init_target_sz = target_sz;
search_area_scale = params.search_area_scale;
max_image_sample_size = params.max_image_sample_size;
max_image_sample_size = params.max_image_sample_size;
min_image_sample_size = params.min_image_sample_size;
refinement_iterations = params.refinement_iterations;
nScales = params.number_of_scales;
scale_step = params.scale_step;
features = params.t_features;

prior_weights = [];
sample_weights = [];
latest_ind = [];

if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end
if ~isfield(params, 'interpolation_method')
    params.interpolation_method = 'none';
end
if ~isfield(params, 'interpolation_centering')
    params.interpolation_centering = false;
end
if ~isfield(params, 'interpolation_windowing')
    params.interpolation_windowing = false;
end
if ~isfield(params, 'clamp_position')
    params.clamp_position = false;
end

if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end
search_area = prod(init_target_sz * search_area_scale);
if search_area > max_image_sample_size
    currentScaleFactor = sqrt(search_area / max_image_sample_size);
elseif search_area < min_image_sample_size
    currentScaleFactor = sqrt(search_area / min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

%window size, taking padding into account
base_target_sz = target_sz / currentScaleFactor;
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor( base_target_sz * search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*4]; % for testing
end
[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'odd_cells');
features{1}.fparams.output_layer=[0 3 3];
features{1}.fparams.net.layers=features{1}.fparams.net.layers(1:3);
img_sample_sz = feature_info.img_sample_sz;
img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;
feature_sz(3,:)=[34 34];
feature_dim = feature_info.dim;
feature_dim=params.compressed_dim(1:length(feature_info.dim))';

reg_window_edge = {};
for k = 1:length(features)
    if isfield(features{k}.fparams, 'reg_window_edge')
        reg_window_edge = cat(3, reg_window_edge, permute(num2cell(features{k}.fparams.reg_window_edge(:)), [2 3 1]));
    else
        reg_window_edge = cat(3, reg_window_edge, cell(1, 1, length(features{k}.fparams.nDim)));
    end
end
[reg_filter, binary_mask, patch_mask] = cellfun(@(reg_window_edge) get_reg_filter(img_support_sz, base_target_sz, params, reg_window_edge), reg_window_edge, 'uniformoutput', false);

num_feature_blocks = length(feature_dim);
% Size of the extracted feature maps

feature_reg = permute(num2cell(feature_info.penalty), [2 3 1]);
filter_sz = feature_sz; 

feature_sz_cell = permute(mat2cell(feature_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

filter_sz = feature_sz + mod(feature_sz+1, 2);
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

set_tracker_param1


for block_id=1:num_feature_blocks
      binary_mask{block_id}=imresize(binary_mask{block_id}, [filter_sz(block_id,1),filter_sz(block_id,2)], 'nearest');
end

[patch_mask]=get_binary_patch_mask(binary_mask,params.frag_num);

output_sz = max(filter_sz, [], 1); 
pad_sz = cellfun(@(filter_sz) (output_sz - filter_sz) / 2, filter_sz_cell, 'uniformoutput', false); 


ky = circshift(-floor((output_sz(1) - 1)/2) : ceil((output_sz(1) - 1)/2), [1, -floor((output_sz(1) - 1)/2)])';
kx = circshift(-floor((output_sz(2) - 1)/2) : ceil((output_sz(2) - 1)/2), [1, -floor((output_sz(2) - 1)/2)]);
ky_tp = ky';
kx_tp = kx';

cos_window = cellfun(@(sz) single(hann(sz(1)+2)*hann(sz(2)+2)'), feature_sz_cell, 'uniformoutput', false);
cos_window = cellfun(@(cos_window) cos_window(2:end-1,2:end-1), cos_window, 'uniformoutput', false);

[interp1_fs, interp2_fs] = cellfun(@(sz) get_interp_fourier(sz, params), filter_sz_cell, 'uniformoutput', false);

%% considers scale
if nScales > 0
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    
    scaleFactors = scale_step .^ scale_exp;
    
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

feature_info.scaleFactors=scaleFactors;
feature_info.target_sz=target_sz;

samplesf = cell(1, 1, num_feature_blocks);
for k = 1:num_feature_blocks
    samplesf{k} = complex(zeros(params.nSamples,feature_dim(k),filter_sz(k,1),(filter_sz(k,2)+1)/2,'single'));
end


for block_id=1:num_feature_blocks
    output_sigma_factor=0.1;
    cell_size=img_sample_sz./filter_sz(block_id,:);
    cell_size=mean(cell_size);
    output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
    yl{block_id} = gaussian_shaped_labels(output_sigma,filter_sz(block_id,:)); 
    yl_fft_gpu{block_id}=fft2(yl{block_id});
end

%% construct yf
sig_y = sqrt(prod(floor(base_target_sz))) * output_sigma_factor * (output_sz ./ img_support_sz);
yf_y = single(sqrt(2*pi) * sig_y(1) / output_sz(1) * exp(-2 * (pi * sig_y(1) * ky / output_sz(1)).^2));
yf_x = single(sqrt(2*pi) * sig_y(2) / output_sz(2) * exp(-2 * (pi * sig_y(2) * kx / output_sz(2)).^2));
y_dft = yf_y * yf_x*100;
yf = cellfun(@(sz) fftshift(resizeDFT2(y_dft, sz, false)), filter_sz_cell, 'uniformoutput', false);
  for block_id=1:num_feature_blocks
        y_real{1,1,block_id}=real(ifft2( ifftshift(ifftshift(yf {1,1,block_id},1),2) ));
     
  end

yf = compact_fourier_coeff(yf);
max_response1=max(max(y_real{1,1,1}));


params.nSamples = min(params.nSamples, numel(seq.s_frames));
for frame=1:seq.endFrame-seq.startFrame+1
       im = imread([seq.path,seq.s_frames(frame).name]);
       
if frame>1
      if size(im,3) > 1 && is_color_image == false
                   im = im(:,:,1);
      end
      
      xt = extract_features(im, pos, currentScaleFactor*scaleFactors, features, global_fparams,binary_mask,patch_mask);
                            
      for scale_ind=1:5
         img_samples(:,:,:,scale_ind) = single(sample_patch(im, pos, round(img_support_sz*currentScaleFactor*scaleFactors(scale_ind)), [271 271]));
      end
                    
      img_samples = impreprocess(img_samples);
      xt{3}=extract_vgg16(img_samples,fsolver,feature_input,feature_blob4,global_fparams);
      clear img_samples;
      xt = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt, cos_window, 'uniformoutput', false);
      xtf = cellfun(@cfft2, xt, 'uniformoutput', false);
      xtf = interpolate_dft(xtf, interp1_fs, interp2_fs);
                
      for block_id=1:num_feature_blocks
          tmp=weight_f{block_id};
          score_fs{1,1,block_id}=bsxfun(@times,conj(tmp), ((xtf{block_id})));
      end
                
      scores_fs_feat = cellfun(@(score_fs, pad_sz) padarray(sum(score_fs, 3), pad_sz), score_fs, pad_sz, 'uniformoutput', false);
      scores_fs=scores_fs_feat{1};
      for block_id=2:num_feature_blocks
           scores_fs=scores_fs+scores_fs_feat{block_id};
      end
                
      scores_fs=permute(scores_fs,[1 2 4 3]);
      scores_fs=ifftshift(ifftshift(scores_fs,1),2);
      newton_iterations=5;
      
      [translation_vec,scale_ind] = optimize_scores(scores_fs, newton_iterations, ky_tp, kx_tp, currentScaleFactor, feature_info, filter_sz);
     
              
            currentScaleFactor = currentScaleFactor * scaleFactors(scale_ind);
            % adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            target_sz = floor(base_target_sz * currentScaleFactor);
            % update position
            old_pos = pos;
            pos = pos + translation_vec;
            
               if frame<10
                   params.learning_rate=0.011;
               else
                  params.learning_rate=0.02;
               end
              
                visualization=1;
                rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
     if visualization == 1
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if frame == 2,  %first frame, create GUI
            fig_handle = figure('Name', 'Tracking');
%             set(fig_handle, 'Position', [100, 100, size(im,2), size(im,1)]);
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(frame), 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        else
            % Do visualization of the sampled confidence scores overlayed
            resp_sz = round(img_support_sz*currentScaleFactor*scaleFactors(scale_ind));
            xs = floor(old_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
            ys = floor(old_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
            
            % To visualize the continuous scores, sample them 10 times more
            % dense than output_sz. This is implemented an an ifft2.
            % First pad the fourier series with zeros.
            sampled_scores_display_dft = resizeDFT2(scores_fs(:,:,scale_ind), 10*output_sz, false);
            % Then do inverse DFT and rescale correctly
            sampled_scores_display = fftshift(prod(10*output_sz) * ifft2(sampled_scores_display_dft, 'symmetric'));
            
            figure(fig_handle);
%                 set(fig_handle, 'Position', [100, 100, 100+size(im,2), 100+size(im,1)]);
            imagesc(im_to_show);
            hold on;
%             resp_handle = imagesc(xs, ys, sampled_scores_display); colormap hsv;
%             alpha(resp_handle, 0.5);
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(frame), 'color', [0 1 1]);
            hold off;
%                imwrite(frame2im(getframe(gcf)),sprintf('result/%04d.bmp',frame));
%                results(frame,:)=rect_position_vis;
%                 axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        end
        
        drawnow
      end   
             results(frame,:)=rect_position_vis;    
                
       end
          %% extract training samples
                if size(im,3) > 1 && is_color_image == false
                   im = im(:,:,1);
                end
                xl = extract_features(im, pos, currentScaleFactor, features, global_fparams,binary_mask,patch_mask);
               
              for scale_ind=1:1
                 img_samples(:,:,:,scale_ind) = single(sample_patch(im, pos, round(img_support_sz*currentScaleFactor), [271 271]));
              end
              
              img_samples = impreprocess(img_samples);
              xl{3}=extract_vgg16(img_samples,fsolver,feature_input,feature_blob4,global_fparams);
              clear img_samples;                         
                
              xl = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
                
                
              xlf = cellfun(@cfft2, xl, 'uniformoutput', false);
                
              xlf = interpolate_dft(xlf, interp1_fs, interp2_fs);
                
              xlf = cellfun(@(xf) xf(:,1:(size(xf,2)+1)/2,:), xlf, 'uniformoutput', false);
    
              model_update;
                           
      end
                
               feature_input.clear_memory_function(1);
               cnna.net.forward_prefilled();
               cnn_third.net.forward_prefilled();













