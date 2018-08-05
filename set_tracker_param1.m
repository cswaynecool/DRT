%% setting caffe parameters
caffe.reset_all();

fid=fopen('data.txt','wt');
if size(filter_sz,1)==5
fprintf(fid,'%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n',filter_sz(1,1),filter_sz(1,2),params.compressed_dim(1),filter_sz(2,1),filter_sz(2,2),params.compressed_dim(2),filter_sz(3,1),filter_sz(3,2),params.compressed_dim(3),filter_sz(4,1),filter_sz(4,2),params.compressed_dim(4),filter_sz(5,1),filter_sz(5,2),params.compressed_dim(5),params.frag_num,params.nSamples);
else
    fprintf(fid,'%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n',filter_sz(1,1),filter_sz(1,2),params.compressed_dim(1),filter_sz(2,1),filter_sz(2,2),params.compressed_dim(2),filter_sz(3,1),filter_sz(3,2),params.compressed_dim(3),filter_sz(4,1),filter_sz(4,2),params.compressed_dim(4),0,0,0,params.frag_num,params.nSamples);
end

fclose(fid);
% % 
% % %% init caffe %%
gpu_id = 0;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
feature_solver_def_file = 'model/feature_solver.prototxt';
model_file = 'model/VGG_ILSVRC_16_layers.caffemodel';

fsolver = caffe.Solver(feature_solver_def_file);
fsolver.net.copy_from(model_file);
feature_input = fsolver.net.blobs('data');
feature_blob4 = fsolver.net.blobs('conv4_3');
% %% spn solver
% 
% % spn_solver_def_file = 'model/spn_solver.prototxt';
% % spn = caffe.Solver(spn_solver_def_file);
% %% cnn-a solver
cnna_solver_def_file = 'model/cnn-a_solver.prototxt'; 
cnna = caffe.Solver(cnna_solver_def_file);
cnn_third_solver_def_file = 'model/cnn-third_solver.prototxt'; 
cnn_third = caffe.Solver(cnn_third_solver_def_file);
% %% cnn-b solver
cnnb_solver_def_file = 'model/cnn-b_solver.prototxt'; 
cnnb = caffe.Solver(cnnb_solver_def_file);
%% cnn-c solver
cnnc_solver_def_file = 'model/cnn-c_solver.prototxt'; 
cnnc = caffe.Solver(cnnc_solver_def_file);
% 
% % % tracker_solver_def_file = 'model/tracker_solver.prototxt'; 
% % % tracker= caffe.Solver(tracker_solver_def_file);



