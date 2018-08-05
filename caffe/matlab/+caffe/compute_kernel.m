function out = compute_kernel(data1,data2)
% set_mode_cpu()
%   set Caffe to CPU mode

out = caffe_('compute_kerenl', data1,data2);

end
