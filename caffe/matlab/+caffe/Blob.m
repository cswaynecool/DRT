classdef Blob < handle
  % Wrapper class of caffe::Blob in matlab
  
  properties (Access = private)
    hBlob_self
  end
  
  methods
    function self = Blob(hBlob_blob)
      CHECK(is_valid_handle(hBlob_blob), 'invalid Blob handle');
      
      % setup self handle
      self.hBlob_self = hBlob_blob;
    end
    function shape = shape(self)
      shape = caffe_('blob_get_shape', self.hBlob_self);
    end
    function reshape(self, shape)
      shape = self.check_and_preprocess_shape(shape);
      caffe_('blob_reshape', self.hBlob_self, shape);
    end
    function data = get_data(self)
      data = caffe_('blob_get_data', self.hBlob_self);
    end
    function set_data(self, data)
      data = self.check_and_preprocess_data(data);
      caffe_('blob_set_data', self.hBlob_self, data);
    end
      function set_data_a(self, data,data1)
%       data = self.check_and_preprocess_data(data);
      caffe_('my_set_data_a', self.hBlob_self, single(data),single(data1));
      end
      
      
      function inialize_blobs(self, net_index,num,channel,height,width,frag_num)
%       data = self.check_and_preprocess_data(data);
      caffe_('inialize_blobs', self.hBlob_self, single(net_index),single(num),single(channel),single(height),single(width),single(frag_num));
      end
      
      function set_data_a1(self, data,data1)
%       data = self.check_and_preprocess_data(data);
      caffe_('my_set_data_a1', self.hBlob_self, single(data),single(data1));
      end
      
        function input_yf(self, data,data1,L_index,num)
%       data = self.check_and_preprocess_data(data);
      caffe_('input_yf', self.hBlob_self, single(data),single(data1),single(L_index),single(num));
      end
      
      function set_data_output_a(self, data,data1)
%       data = self.check_and_preprocess_data(data);
      caffe_('my_set_output_a', self.hBlob_self, single(data),single(data1));
      end
      
function get_data_from_matlab(self, hf_real,hf_imag,net_index)
%       data = self.check_and_preprocess_data(data);
      caffe_('get_data_from_matlab', self.hBlob_self, single(hf_real),single(hf_imag),single(net_index));
end
      
function update_samplesf(self, samplef_real,samplef_imag,net_index,replace_id)
%       data = self.check_and_preprocess_data(data);
          caffe_('update_samplesf', self.hBlob_self, single(samplef_real),single(samplef_imag),single(net_index),single(replace_id));
end

function input_sample_weight(self, weight)
%       data = self.check_and_preprocess_data(data);
          caffe_('input_sample_weight', self.hBlob_self, single(weight));
end
     
function  col_im=im2col(self, data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w)
%       data = self.check_and_preprocess_data(data);
          col_im=caffe_('im2col', self.hBlob_self, single(data), single(kernel_h), single(kernel_w), single(stride_h), single(stride_w), single(pad_h), single(pad_w));
end

function set_reg_window(self, fftshift_mask1,fftshift_mask2,ifftshift_mask1,ifftshift_mask2,binary_mask1,binary_mask2)
%       data = self.check_and_preprocess_data(data);
          caffe_('set_reg_window', self.hBlob_self, single(fftshift_mask1),single(fftshift_mask2),single(ifftshift_mask1),single(ifftshift_mask2),single(binary_mask1),single(binary_mask2));
end

function set_index(self, index,index1,total_num,total_num1,total_num2,total_num3)
%       data = self.check_and_preprocess_data(data);
          caffe_('set_index', self.hBlob_self, single(index),single(index1),single(total_num),single(total_num1),single(total_num2),single(total_num3));
end

function set_hf_5(self, hf_real1, hf_imag1,  hf_real2, hf_imag2, hf_real3, hf_imag3, hf_real4,hf_imag4,hf_real5,hf_imag5)
%       data = self.check_and_preprocess_data(data);
          caffe_('set_hf_5', self.hBlob_self, single(hf_real1), single(hf_imag1),  single(hf_real2), single(hf_imag2), single(hf_real3), single(hf_imag3),...
              single(hf_real4),single(hf_imag4),single(hf_real5),single(hf_imag5));
end

function set_hf_4(self, hf_real1, hf_imag1,  hf_real2, hf_imag2, hf_real3, hf_imag3, hf_real4,hf_imag4)
%       data = self.check_and_preprocess_data(data);
          caffe_('set_hf_4', self.hBlob_self, single(hf_real1), single(hf_imag1),  single(hf_real2), single(hf_imag2), single(hf_real3), single(hf_imag3),...
              single(hf_real4),single(hf_imag4));
end

function set_L_index(self, L_index,L_index1)
%       data = self.check_and_preprocess_data(data);
          caffe_('set_L_index', self.hBlob_self, single(L_index),single(L_index1));
end

function set_frame_id(self, frame)
%       data = self.check_and_preprocess_data(data);
          caffe_('set_frame_id', self.hBlob_self, single(frame));
end

function set_H_5(self, H1,H2,H3,H4,H5,frag_num)
%       data = self.check_and_preprocess_data(data);
          caffe_('set_H_5', self.hBlob_self, single(H1), single(H2),  single(H3), single(H4), single(H5),single(frag_num));
end

function set_H_4(self, H1,H2,H3,H4,frag_num)
%       data = self.check_and_preprocess_data(data);
          caffe_('set_H_4', self.hBlob_self, single(H1), single(H2),  single(H3), single(H4),single(frag_num));
end

function set_binary_mask_adaptive(self, binary_mask1,binary_mask2)
%       data = self.check_and_preprocess_data(data);
          caffe_('set_binary_mask_adaptive', self.hBlob_self, single(binary_mask1),single(binary_mask2));
end

function set_patch_mask(self, patch_mask1, patch_mask2)
%       data = self.check_and_preprocess_data(data);
          caffe_('set_patch_mask', self.hBlob_self, single(patch_mask1), single(patch_mask2));
end

%         function set_momentum(self, momentum)
% %       data = self.check_and_preprocess_data(data);
%       caffe_('set_momentum', self.hBlob_self, single(momentum));
%         end
     
      function [data1,data2,data3,data4,data5,data6,data7] = my_get_data(self)
      [data1,data2,data3,data4,data5,data6,data7] = caffe_('my_get_data', self.hBlob_self);
      end
      
      function  dt_pooling_parameter(self,height,width,x,x1,y,y1)
       caffe_('dt_pooling_parameter', self.hBlob_self,single(height),single(width),single(x),single(x1),single(y),single(y1));
      end
    
       function  [data1,data2]=get_tmp(self)
       [data1,data2]=caffe_('get_tmp', self.hBlob_self);
       end
    
         function  [data1,data2]=get_samplesf(self,net_index)
       [data1,data2]=caffe_('get_samplesf', self.hBlob_self,single(net_index));
         end
      
      function  set_xf_yf4(self,x_real1,x_imag1,x_real2,x_imag2,x_real3,x_imag3,x_real4,x_imag4,y_real1,y_imag1,y_real2,y_imag2,y_real3,y_imag3,y_real4,y_imag4)
       caffe_('set_xf_yf4', self.hBlob_self,single(x_real1),single(x_imag1),single(x_real2),single(x_imag2),single(x_real3),single(x_imag3),single(x_real4),single(x_imag4),...
       single(y_real1),single(y_imag1),single(y_real2),single(y_imag2),single(y_real3),single(y_imag3),single(y_real4),single(y_imag4)    );
      end
    
           function  set_xf_yf5(self,x_real1,x_imag1,x_real2,x_imag2,x_real3,x_imag3,x_real4,x_imag4,x_real5,x_imag5,y_real1,y_imag1,y_real2,y_imag2,y_real3,y_imag3,y_real4,y_imag4,y_real5,y_imag5 )
       caffe_('set_xf_yf5', self.hBlob_self,single(x_real1),single(x_imag1),single(x_real2),single(x_imag2),single(x_real3),single(x_imag3),single(x_real4),single(x_imag4),single(x_real5),single(x_imag5),...
          single(y_real1),single(y_imag1),single(y_real2),single(y_imag2),single(y_real3),single(y_imag3),single(y_real4),single(y_imag4),single(y_real5),single(y_imag5) );
           end
    
    function  clear_memory_function(self,clear_memory)
        caffe_('clear_memory_function', self.hBlob_self,single(clear_memory));
       end
           
    function diff = get_diff(self)
      diff = caffe_('blob_get_diff', self.hBlob_self);
    end
    function set_diff(self, diff)
      diff = self.check_and_preprocess_data(diff);
      caffe_('blob_set_diff', self.hBlob_self, diff);
    end
  end
  
  methods (Access = private)
    function shape = check_and_preprocess_shape(~, shape)
      CHECK(isempty(shape) || (isnumeric(shape) && isrow(shape)), ...
        'shape must be a integer row vector');
      shape = double(shape);
    end
    function data = check_and_preprocess_data(self, data)
      CHECK(isnumeric(data), 'data or diff must be numeric types');
      self.check_data_size_matches(data);
      if ~isa(data, 'single')
        data = single(data);
      end
    end
    function check_data_size_matches(self, data)
      % check whether size of data matches shape of this blob
      % note: matlab arrays always have at least 2 dimensions. To compare
      % shape between size of data and shape of this blob, extend shape of
      % this blob to have at least 2 dimensions
      self_shape_extended = self.shape;
      if isempty(self_shape_extended)
        % target blob is a scalar (0 dim)
        self_shape_extended = [1, 1];
      elseif isscalar(self_shape_extended)
        % target blob is a vector (1 dim)
        self_shape_extended = [self_shape_extended, 1];
      end
      % Also, matlab cannot have tailing dimension 1 for ndim > 2, so you
      % cannot create 20 x 10 x 1 x 1 array in matlab as it becomes 20 x 10
      % Extend matlab arrays to have tailing dimension 1 during shape match
      data_size_extended = ...
        [size(data), ones(1, length(self_shape_extended) - ndims(data))];
      is_matched = ...
        (length(self_shape_extended) == length(data_size_extended)) ...
        && all(self_shape_extended == data_size_extended);
      CHECK(is_matched, ...
        sprintf('%s, input data/diff size: [ %s] vs target blob shape: [ %s]', ...
        'input data/diff size does not match target blob shape', ...
        sprintf('%d ', data_size_extended), sprintf('%d ', self_shape_extended)));
    end
  end
end
