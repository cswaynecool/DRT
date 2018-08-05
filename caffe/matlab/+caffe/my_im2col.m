function out = my_im2col(im, k_h, k_w, s_h, s_w,pad_h,pad_w)
% set_mode_cpu()
%   set Caffe to CPU mode

out = caffe_('im2col', im, k_h, k_w, s_h, s_w, pad_h,pad_w);

end
