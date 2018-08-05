close all
% cd ./trackers/FCNT/
% addpath('caffe-fcnt/matlab/caffe/','util/');
seq_name = 'bolt';
init_rect=load([seq_name,'/groundtruth_rect.txt']);
seq.init_rect = init_rect(1,:);
seq.startFrame=1;
seq.endFrame=size(init_rect,1);
seq.path=[seq_name,'/img/'];
seq.s_frames=dir([ [seq_name,'/img/'],'*.jpg']);
res = DRT(seq);
% res=res.res;
caffe.reset_all();