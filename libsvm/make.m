% This make.m is used under Windows

% % add -largeArrayDims on 64-bit machines
% mex -O -c svm.cpp
% mex -O -c svm_model_matlab.c
% mex -O svmtrain.c svm.obj svm_model_matlab.obj
% mex -O svmpredict.c svm.obj svm_model_matlab.obj
% mex -O libsvmread.c
% mex -O libsvmwrite.c

% for 64-bit machine
mex  -largeArrayDims -O -c svm.cpp
mex  -largeArrayDims -O -c svm_model_matlab.c
mex  -largeArrayDims -O svmtrain.c svm.obj svm_model_matlab.obj
mex  -largeArrayDims -O svmpredict.c svm.obj svm_model_matlab.obj
mex  -largeArrayDims -O libsvmread.c
mex  -largeArrayDims -O libsvmwrite.c

