//
// caffe_.cpp provides wrappers of the caffe::Solver class, caffe::Net class,
// caffe::Layer class and caffe::Blob class and some caffe::Caffe functions,
// so that one could easily use Caffe from matlab.
// Note that for matlab, we will simply use float as the data type.

// Internally, data is stored with dimensions reversed from Caffe's:
// e.g., if the Caffe blob axes are (num, channels, height, width),
// the matcaffe data is stored as (width, height, channels, num)
// where width is the fastest dimension.

#include <sstream>
#include <string>
#include <vector>
#include "caffe/util/math_functions.hpp"
#include "mex.h"

#include "caffe/caffe.hpp"
#include "caffe/util/im2col.hpp"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

using namespace caffe;  // NOLINT(build/namespaces)

// Do CHECK and throw a Mex error if check fails
inline void mxCHECK(bool expr, const char* msg) {
  if (!expr) {
    mexErrMsgTxt(msg);
  }
}
inline void mxERROR(const char* msg) { mexErrMsgTxt(msg); }

// Check if a file exists and can be opened
void mxCHECK_FILE_EXIST(const char* file) {
  std::ifstream f(file);
  if (!f.good()) {
    f.close();
    std::string msg("Could not open file ");
    msg += file;
    mxERROR(msg.c_str());
  }
  f.close();
}

// The pointers to caffe::Solver and caffe::Net instances
static vector<shared_ptr<Solver<float> > > solvers_;
static vector<shared_ptr<Net<float> > > nets_;
// init_key is generated at the beginning and everytime you call reset
static double init_key = static_cast<double>(caffe_rng_rand());

/** -----------------------------------------------------------------
 ** data conversion functions
 **/
// Enum indicates which blob memory to use
enum WhichMemory { DATA, DIFF };

// Copy matlab array to Blob data or diff
static void mx_mat_to_blob(const mxArray* mx_mat, Blob<float>* blob,
    WhichMemory data_or_diff) {
  mxCHECK(blob->count() == mxGetNumberOfElements(mx_mat),
      "number of elements in target blob doesn't match that in input mxArray");
  const float* mat_mem_ptr = reinterpret_cast<const float*>(mxGetData(mx_mat));
  float* blob_mem_ptr = NULL;
  switch (Caffe::mode()) {
  case Caffe::CPU:
    blob_mem_ptr = (data_or_diff == DATA ?
        blob->mutable_cpu_data() : blob->mutable_cpu_diff());
    break;
  case Caffe::GPU:
    blob_mem_ptr = (data_or_diff == DATA ?
        blob->mutable_gpu_data() : blob->mutable_gpu_diff());
    break;
  default:
    mxERROR("Unknown Caffe mode");
  }
  caffe_copy(blob->count(), mat_mem_ptr, blob_mem_ptr);
}

// Copy Blob data or diff to matlab array
static mxArray* blob_to_mx_mat(const Blob<float>* blob,
    WhichMemory data_or_diff) {
  const int num_axes = blob->num_axes();
  vector<mwSize> dims(num_axes);
  for (int blob_axis = 0, mat_axis = num_axes - 1; blob_axis < num_axes;
       ++blob_axis, --mat_axis) {
    dims[mat_axis] = static_cast<mwSize>(blob->shape(blob_axis));
  }
  // matlab array needs to have at least one dimension, convert scalar to 1-dim
  if (num_axes == 0) {
    dims.push_back(1);
  }
  mxArray* mx_mat =
      mxCreateNumericArray(dims.size(), dims.data(), mxSINGLE_CLASS, mxREAL);
  float* mat_mem_ptr = reinterpret_cast<float*>(mxGetData(mx_mat));
  const float* blob_mem_ptr = NULL;
  switch (Caffe::mode()) {
  case Caffe::CPU:
    blob_mem_ptr = (data_or_diff == DATA ? blob->cpu_data() : blob->cpu_diff());
    break;
  case Caffe::GPU:
    blob_mem_ptr = (data_or_diff == DATA ? blob->gpu_data() : blob->gpu_diff());
    break;
  default:
    mxERROR("Unknown Caffe mode");
  }
  caffe_copy(blob->count(), blob_mem_ptr, mat_mem_ptr);
  return mx_mat;
}

// Convert vector<int> to matlab row vector
static mxArray* int_vec_to_mx_vec(const vector<int>& int_vec) {
  mxArray* mx_vec = mxCreateDoubleMatrix(int_vec.size(), 1, mxREAL);
  double* vec_mem_ptr = mxGetPr(mx_vec);
  for (int i = 0; i < int_vec.size(); i++) {
    vec_mem_ptr[i] = static_cast<double>(int_vec[i]);
  }
  return mx_vec;
}

// Convert vector<string> to matlab cell vector of strings
static mxArray* str_vec_to_mx_strcell(const vector<std::string>& str_vec) {
  mxArray* mx_strcell = mxCreateCellMatrix(str_vec.size(), 1);
  for (int i = 0; i < str_vec.size(); i++) {
    mxSetCell(mx_strcell, i, mxCreateString(str_vec[i].c_str()));
  }
  return mx_strcell;
}

/** -----------------------------------------------------------------
 ** handle and pointer conversion functions
 ** a handle is a struct array with the following fields
 **   (uint64) ptr      : the pointer to the C++ object
 **   (double) init_key : caffe initialization key
 **/
// Convert a handle in matlab to a pointer in C++. Check if init_key matches
template <typename T>
static T* handle_to_ptr(const mxArray* mx_handle) {
  mxArray* mx_ptr = mxGetField(mx_handle, 0, "ptr");
  mxArray* mx_init_key = mxGetField(mx_handle, 0, "init_key");
  mxCHECK(mxIsUint64(mx_ptr), "pointer type must be uint64");
  mxCHECK(mxGetScalar(mx_init_key) == init_key,
      "Could not convert handle to pointer due to invalid init_key. "
      "The object might have been cleared.");
  return reinterpret_cast<T*>(*reinterpret_cast<uint64_t*>(mxGetData(mx_ptr)));
}

// Create a handle struct vector, without setting up each handle in it
template <typename T>
static mxArray* create_handle_vec(int ptr_num) {
  const int handle_field_num = 2;
  const char* handle_fields[handle_field_num] = { "ptr", "init_key" };
  return mxCreateStructMatrix(ptr_num, 1, handle_field_num, handle_fields);
}

// Set up a handle in a handle struct vector by its index
template <typename T>
static void setup_handle(const T* ptr, int index, mxArray* mx_handle_vec) {
  mxArray* mx_ptr = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
  *reinterpret_cast<uint64_t*>(mxGetData(mx_ptr)) =
      reinterpret_cast<uint64_t>(ptr);
  mxSetField(mx_handle_vec, index, "ptr", mx_ptr);
  mxSetField(mx_handle_vec, index, "init_key", mxCreateDoubleScalar(init_key));
}

// Convert a pointer in C++ to a handle in matlab
template <typename T>
static mxArray* ptr_to_handle(const T* ptr) {
  mxArray* mx_handle = create_handle_vec<T>(1);
  setup_handle(ptr, 0, mx_handle);
  return mx_handle;
}

// Convert a vector of shared_ptr in C++ to handle struct vector
template <typename T>
static mxArray* ptr_vec_to_handle_vec(const vector<shared_ptr<T> >& ptr_vec) {
  mxArray* mx_handle_vec = create_handle_vec<T>(ptr_vec.size());
  for (int i = 0; i < ptr_vec.size(); i++) {
    setup_handle(ptr_vec[i].get(), i, mx_handle_vec);
  }
  return mx_handle_vec;
}

/** -----------------------------------------------------------------
 ** matlab command functions: caffe_(api_command, arg1, arg2, ...)
 **/
// Usage: caffe_('get_solver', solver_file);
static void get_solver(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsChar(prhs[0]),
      "Usage: caffe_('get_solver', solver_file)");
  char* solver_file = mxArrayToString(prhs[0]);
  mxCHECK_FILE_EXIST(solver_file);
  shared_ptr<Solver<float> > solver(new caffe::SGDSolver<float>(solver_file));
  solvers_.push_back(solver);
  plhs[0] = ptr_to_handle<Solver<float> >(solver.get());
  mxFree(solver_file);
}

// Usage: caffe_('solver_get_attr', hSolver)
static void solver_get_attr(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('solver_get_attr', hSolver)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  const int solver_attr_num = 2;
  const char* solver_attrs[solver_attr_num] = { "hNet_net", "hNet_test_nets" };
  mxArray* mx_solver_attr = mxCreateStructMatrix(1, 1, solver_attr_num,
      solver_attrs);
  mxSetField(mx_solver_attr, 0, "hNet_net",
      ptr_to_handle<Net<float> >(solver->net().get()));
  mxSetField(mx_solver_attr, 0, "hNet_test_nets",
      ptr_vec_to_handle_vec<Net<float> >(solver->test_nets()));
  plhs[0] = mx_solver_attr;
}

// Usage: caffe_('solver_get_iter', hSolver)
static void solver_get_iter(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('solver_get_iter', hSolver)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  plhs[0] = mxCreateDoubleScalar(solver->iter());
}

// Usage: caffe_('solver_restore', hSolver, snapshot_file)
static void solver_restore(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsChar(prhs[1]),
      "Usage: caffe_('solver_restore', hSolver, snapshot_file)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  char* snapshot_file = mxArrayToString(prhs[1]);
  mxCHECK_FILE_EXIST(snapshot_file);
  solver->Restore(snapshot_file);
  mxFree(snapshot_file);
}

// Usage: caffe_('solver_solve', hSolver)
static void solver_solve(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('solver_solve', hSolver)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  solver->Solve();
}

// Usage: caffe_('solver_step', hSolver, iters)
static void solver_step(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsDouble(prhs[1]),
      "Usage: caffe_('solver_step', hSolver, iters)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  int iters = mxGetScalar(prhs[1]);
  solver->Step(iters);
}

// Usage: caffe_('get_net', model_file, phase_name)
static void get_net(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsChar(prhs[0]) && mxIsChar(prhs[1]),
      "Usage: caffe_('get_net', model_file, phase_name)");
  char* model_file = mxArrayToString(prhs[0]);
  char* phase_name = mxArrayToString(prhs[1]);
  mxCHECK_FILE_EXIST(model_file);
  Phase phase;
  if (strcmp(phase_name, "train") == 0) {
      phase = TRAIN;
  } else if (strcmp(phase_name, "test") == 0) {
      phase = TEST;
  } else {
    mxERROR("Unknown phase");
  }
  shared_ptr<Net<float> > net(new caffe::Net<float>(model_file, phase));
  nets_.push_back(net);
  plhs[0] = ptr_to_handle<Net<float> >(net.get());
  mxFree(model_file);
  mxFree(phase_name);
}

// Usage: caffe_('net_get_attr', hNet)
static void net_get_attr(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('net_get_attr', hNet)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  const int net_attr_num = 6;
  const char* net_attrs[net_attr_num] = { "hLayer_layers", "hBlob_blobs",
      "input_blob_indices", "output_blob_indices", "layer_names", "blob_names"};
  mxArray* mx_net_attr = mxCreateStructMatrix(1, 1, net_attr_num,
      net_attrs);
  mxSetField(mx_net_attr, 0, "hLayer_layers",
      ptr_vec_to_handle_vec<Layer<float> >(net->layers()));
  mxSetField(mx_net_attr, 0, "hBlob_blobs",
      ptr_vec_to_handle_vec<Blob<float> >(net->blobs()));
  mxSetField(mx_net_attr, 0, "input_blob_indices",
      int_vec_to_mx_vec(net->input_blob_indices()));
  mxSetField(mx_net_attr, 0, "output_blob_indices",
      int_vec_to_mx_vec(net->output_blob_indices()));
  mxSetField(mx_net_attr, 0, "layer_names",
      str_vec_to_mx_strcell(net->layer_names()));
  mxSetField(mx_net_attr, 0, "blob_names",
      str_vec_to_mx_strcell(net->blob_names()));
  plhs[0] = mx_net_attr;
}

// Usage: caffe_('net_forward', hNet)
static void net_forward(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('net_forward', hNet)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  net->ForwardPrefilled();
}

// Usage: caffe_('net_backward', hNet)
static void net_backward(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('net_backward', hNet)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  net->Backward();
  //LOG(INFO) << "end of matcaffe net_backward";
}

// Usage: caffe_('net_copy_from', hNet, weights_file)
static void net_copy_from(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsChar(prhs[1]),
      "Usage: caffe_('net_copy_from', hNet, weights_file)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  char* weights_file = mxArrayToString(prhs[1]);
  mxCHECK_FILE_EXIST(weights_file);
  net->CopyTrainedLayersFrom(weights_file);
  mxFree(weights_file);
}

// Usage: caffe_('net_reshape', hNet)
static void net_reshape(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('net_reshape', hNet)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  net->Reshape();
}

// Usage: caffe_('net_save', hNet, save_file)
static void net_save(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsChar(prhs[1]),
      "Usage: caffe_('net_save', hNet, save_file)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  char* weights_file = mxArrayToString(prhs[1]);
  NetParameter net_param;
  net->ToProto(&net_param, false);
  WriteProtoToBinaryFile(net_param, weights_file);
  mxFree(weights_file);
}

// Usage: caffe_('layer_get_attr', hLayer)
static void layer_get_attr(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('layer_get_attr', hLayer)");
  Layer<float>* layer = handle_to_ptr<Layer<float> >(prhs[0]);
  const int layer_attr_num = 1;
  const char* layer_attrs[layer_attr_num] = { "hBlob_blobs" };
  mxArray* mx_layer_attr = mxCreateStructMatrix(1, 1, layer_attr_num,
      layer_attrs);
  mxSetField(mx_layer_attr, 0, "hBlob_blobs",
      ptr_vec_to_handle_vec<Blob<float> >(layer->blobs()));
  plhs[0] = mx_layer_attr;
}

// Usage: caffe_('layer_get_type', hLayer)
static void layer_get_type(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('layer_get_type', hLayer)");
  Layer<float>* layer = handle_to_ptr<Layer<float> >(prhs[0]);
  plhs[0] = mxCreateString(layer->type());
}

// Usage: caffe_('blob_get_shape', hBlob)
static void blob_get_shape(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('blob_get_shape', hBlob)");
  Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
  const int num_axes = blob->num_axes();
  mxArray* mx_shape = mxCreateDoubleMatrix(1, num_axes, mxREAL);
  double* shape_mem_mtr = mxGetPr(mx_shape);
  for (int blob_axis = 0, mat_axis = num_axes - 1; blob_axis < num_axes;
       ++blob_axis, --mat_axis) {
    shape_mem_mtr[mat_axis] = static_cast<double>(blob->shape(blob_axis));
  }
  plhs[0] = mx_shape;
}

// Usage: caffe_('blob_reshape', hBlob, new_shape)
static void blob_reshape(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsDouble(prhs[1]),
      "Usage: caffe_('blob_reshape', hBlob, new_shape)");
  Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
  const mxArray* mx_shape = prhs[1];
  double* shape_mem_mtr = mxGetPr(mx_shape);
  const int num_axes = mxGetNumberOfElements(mx_shape);
  vector<int> blob_shape(num_axes);
  for (int blob_axis = 0, mat_axis = num_axes - 1; blob_axis < num_axes;
       ++blob_axis, --mat_axis) {
    blob_shape[blob_axis] = static_cast<int>(shape_mem_mtr[mat_axis]);
  }
  blob->Reshape(blob_shape);
}

// Usage: caffe_('blob_get_data', hBlob)
static void blob_get_data(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('blob_get_data', hBlob)");
  Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
  plhs[0] = blob_to_mx_mat(blob, DATA);
}

// Usage: caffe_('blob_set_data', hBlob, new_data)
static void blob_set_data(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
      "Usage: caffe_('blob_set_data', hBlob, new_data)");
  Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
  mx_mat_to_blob(prhs[1], blob, DATA);
}


 static void my_set_data_a(MEX_ARGS) {
   mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
  const int* dims = static_cast<const int*>(mxGetDimensions(prhs[1]));  
  Blob<float>* tmp=new Blob<float>(); //存储fft的实部
  tmp->Reshape(1,dims[2],dims[1],dims[0]);
  Blob<float>* tmp1=new Blob<float>(); //存储fft的虚部
  tmp1->Reshape(1,dims[2],dims[1],dims[0]);
  mx_mat_to_blob(prhs[1],tmp, DATA);
  mx_mat_to_blob(prhs[2],tmp1, DATA); 
  Layer<float>::first_layer_fft_real.resize(1);
  Layer<float>::first_layer_fft_real[0].reset(new Blob<float>());
  Layer<float>::first_layer_fft_real[0]->Reshape(1,dims[2],dims[1],dims[0]);
  
  Layer<float>::first_layer_fft_imag.resize(1);
  Layer<float>::first_layer_fft_imag[0].reset(new Blob<float>());
  Layer<float>::first_layer_fft_imag[0]->Reshape(1,dims[2],dims[1],dims[0]); 

  caffe_copy(tmp->count(),tmp->mutable_gpu_data(),Layer<float>::first_layer_fft_real[0]->mutable_gpu_data());
  caffe_copy(tmp1->count(),tmp1->mutable_gpu_data(),Layer<float>::first_layer_fft_imag[0]->mutable_gpu_data());
  delete tmp;
  delete tmp1;
 }

static void my_set_data_a1(MEX_ARGS) {
   mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
  const int* dims = static_cast<const int*>(mxGetDimensions(prhs[1]));  
  Blob<float>* tmp=new Blob<float>(); //存储fft的实部
  tmp->Reshape(1,dims[2],dims[1],dims[0]);
  Blob<float>* tmp1=new Blob<float>(); //存储fft的虚部
  tmp1->Reshape(1,dims[2],dims[1],dims[0]);
  mx_mat_to_blob(prhs[1],tmp, DATA);
  mx_mat_to_blob(prhs[2],tmp1, DATA); 
  Layer<float>::second_layer_fft_real.resize(1);
  Layer<float>::second_layer_fft_real[0].reset(new Blob<float>());
  Layer<float>::second_layer_fft_real[0]->Reshape(1,dims[2],dims[1],dims[0]);
  
  Layer<float>::second_layer_fft_imag.resize(1);
  Layer<float>::second_layer_fft_imag[0].reset(new Blob<float>());
  Layer<float>::second_layer_fft_imag[0]->Reshape(1,dims[2],dims[1],dims[0]); 

  caffe_copy(tmp->count(),tmp->mutable_gpu_data(),Layer<float>::second_layer_fft_real[0]->mutable_gpu_data());
  caffe_copy(tmp1->count(),tmp1->mutable_gpu_data(),Layer<float>::second_layer_fft_imag[0]->mutable_gpu_data());
  delete tmp;
  delete tmp1;
 }

static void my_set_output_a(MEX_ARGS) {
   mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
  const int* dims = static_cast<const int*>(mxGetDimensions(prhs[1]));  
  Blob<float>* tmp=new Blob<float>(); //存储fft的实部
  tmp->Reshape(1,dims[2],dims[1],dims[0]);
  Blob<float>* tmp1=new Blob<float>(); //存储fft的虚部
  tmp1->Reshape(1,dims[2],dims[1],dims[0]);
  mx_mat_to_blob(prhs[1],tmp, DATA);
  mx_mat_to_blob(prhs[2],tmp1, DATA); 
  Layer<float>::neta_out_fft_real.resize(1);
  Layer<float>::neta_out_fft_real[0].reset(new Blob<float>());
  Layer<float>::neta_out_fft_real[0]->Reshape(1,dims[2],dims[1],dims[0]);
  
  Layer<float>::neta_out_fft_imag.resize(1);
  Layer<float>::neta_out_fft_imag[0].reset(new Blob<float>());
  Layer<float>::neta_out_fft_imag[0]->Reshape(1,dims[2],dims[1],dims[0]);
 
  Layer<float>::neta_loss_fft_real.resize(1);
  Layer<float>::neta_loss_fft_real[0].reset(new Blob<float>());
  Layer<float>::neta_loss_fft_real[0]->Reshape(1,dims[2],dims[1],dims[0]);
  
  Layer<float>::neta_loss_fft_imag.resize(1);
  Layer<float>::neta_loss_fft_imag[0].reset(new Blob<float>());
  Layer<float>::neta_loss_fft_imag[0]->Reshape(1,dims[2],dims[1],dims[0]); 


  caffe_copy(tmp->count(),tmp->mutable_gpu_data(),Layer<float>::neta_out_fft_real[0]->mutable_gpu_data());
  caffe_copy(tmp1->count(),tmp1->mutable_gpu_data(),Layer<float>::neta_out_fft_imag[0]->mutable_gpu_data());
  delete tmp;
  delete tmp1;
}

static void inialize_blobs(MEX_ARGS) {
   
    const  mxArray* mx_mat1=prhs[1];
    const float* net_index = reinterpret_cast<const float*>(mxGetData(mx_mat1));
    const  mxArray* mx_mat2=prhs[2];
    const float* num = reinterpret_cast<const float*>(mxGetData(mx_mat2));
    const  mxArray* mx_mat3=prhs[3];
    const float* channel = reinterpret_cast<const float*>(mxGetData(mx_mat3));
    const  mxArray* mx_mat4=prhs[4];
    const float* height = reinterpret_cast<const float*>(mxGetData(mx_mat4));
    const  mxArray* mx_mat5=prhs[5];
    const float* width = reinterpret_cast<const float*>(mxGetData(mx_mat5));
    const  mxArray* mx_mat6=prhs[6];
    const float* frag_num = reinterpret_cast<const float*>(mxGetData(mx_mat6));

    Layer<float>::clear_memory.resize(1);
    Layer<float>::clear_memory[0].reset(new Blob<float>());
    Layer<float>::clear_memory[0]->Reshape(1,1,1,1);

    if(net_index[0]==1)
    {

       Layer<float>::reliability_response.resize(1);
       Layer<float>::reliability_response[0].reset(new Blob<float>());
       Layer<float>::reliability_response[0]->Reshape(frag_num[0],num[0],height[0],width[0]);
 
       Layer<float>::reliability_response1.resize(1);
       Layer<float>::reliability_response1[0].reset(new Blob<float>());
       Layer<float>::reliability_response1[0]->Reshape(frag_num[0],num[0],height[0],width[0]);
 
       Layer<float>::reliability_response2.resize(1);
       Layer<float>::reliability_response2[0].reset(new Blob<float>());
       Layer<float>::reliability_response2[0]->Reshape(frag_num[0],num[0],height[0],width[0]);

      Layer<float>::sh_real.resize(1);
      Layer<float>::sh_real[0].reset(new Blob<float>());
      Layer<float>::sh_real[0]->Reshape(num[0],1,height[0],width[0]/2+1);

      Layer<float>::sh_imag.resize(1);
      Layer<float>::sh_imag[0].reset(new Blob<float>());
      Layer<float>::sh_imag[0]->Reshape(num[0],1,height[0],width[0]/2+1);

      Layer<float>::feature_num.resize(1);
      Layer<float>::feature_num[0].reset(new Blob<float>());
      Layer<float>::feature_num[0]->Reshape(1,1,1,1);  
     // printf("we come here no problem\n\n");  
      Layer<float>::first_layer_hf_real.resize(5);
      Layer<float>::first_layer_hf_imag.resize(5);

      Layer<float>::hf1.resize(5);

      Layer<float>::hf_tmp_real.resize(5);
      Layer<float>::hf_tmp_imag.resize(5);   

      Layer<float>::matlab_hf_real.resize(5);
      Layer<float>::matlab_hf_imag.resize(5);      

      Layer<float>::input_xff_real.resize(5);
      Layer<float>::input_xff_imag.resize(5);
      Layer<float>::input_yff_real.resize(5);
      Layer<float>::input_yff_imag.resize(5);   

      Layer<float>::KK_real.resize(5);
      Layer<float>::KK_imag.resize(5);  

      Layer<float>::laplace_real.resize(5);
      Layer<float>::laplace_imag.resize(5);

      Layer<float>::hf1[0].reset(new Blob<float>());
      Layer<float>::hf1[0]->Reshape(1,channel[0],height[0],width[0]/2+1);

      Layer<float>::laplace_real[0].reset(new Blob<float>());
      Layer<float>::laplace_real[0]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::laplace_imag[0].reset(new Blob<float>());
      Layer<float>::laplace_imag[0]->Reshape(1,channel[0],height[0],width[0]/2+1); 


      Layer<float>::first_layer_tmp_real1.resize(5);
      Layer<float>::first_layer_tmp_imag1.resize(5);

      Layer<float>::first_layer_samplef_real.resize(5);
      Layer<float>::first_layer_samplef_imag.resize(5);
      Layer<float>::first_layer_weighted_sample_real.resize(5);
      Layer<float>::first_layer_weighted_sample_imag.resize(5);
      Layer<float>::first_layer_weighted_sample_real1.resize(5);
      Layer<float>::first_layer_weighted_sample_imag1.resize(5);
   
      Layer<float>::H_total.resize(5);
      Layer<float>::filter_H.resize(5); 

      Layer<float>::second_layer_hf_real.resize(5);
      Layer<float>::second_layer_hf_imag.resize(5);  

      Layer<float>::second_layer_weighted_sample_real.resize(5);
      Layer<float>::second_layer_weighted_sample_imag.resize(5);

      Layer<float>::KK_real[0].reset(new Blob<float>());
      Layer<float>::KK_imag[0].reset(new Blob<float>());
     
      Layer<float>::KK_real[0]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::KK_imag[0]->Reshape(1,channel[0],height[0],width[0]/2+1);

      Layer<float>::first_layer_hf_real[0].reset(new Blob<float>());
      Layer<float>::first_layer_hf_real[0]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::first_layer_hf_imag[0].reset(new Blob<float>());
      Layer<float>::first_layer_hf_imag[0]->Reshape(1,channel[0],height[0],width[0]/2+1);

      Layer<float>::input_xff_real[0].reset(new Blob<float>());
      Layer<float>::input_xff_real[0]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::input_xff_imag[0].reset(new Blob<float>());
      Layer<float>::input_xff_imag[0]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::input_yff_real[0].reset(new Blob<float>());
      Layer<float>::input_yff_real[0]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::input_yff_imag[0].reset(new Blob<float>());
      Layer<float>::input_yff_imag[0]->Reshape(1,channel[0],height[0],width[0]/2+1);  


      Layer<float>::hf_tmp_real[0].reset(new Blob<float>());
      Layer<float>::hf_tmp_real[0]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::hf_tmp_imag[0].reset(new Blob<float>());
      Layer<float>::hf_tmp_imag[0]->Reshape(1,channel[0],height[0],width[0]/2+1);

      Layer<float>::matlab_hf_real[0].reset(new Blob<float>());
      Layer<float>::matlab_hf_real[0]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::matlab_hf_imag[0].reset(new Blob<float>());
      Layer<float>::matlab_hf_imag[0]->Reshape(1,channel[0],height[0],width[0]/2+1); 


      Layer<float>::first_layer_samplef_real[0].reset(new Blob<float>());
      Layer<float>::first_layer_samplef_real[0]->Reshape(num[0],channel[0],height[0],width[0]/2+1);
      Layer<float>::first_layer_samplef_imag[0].reset(new Blob<float>());
      Layer<float>::first_layer_samplef_imag[0]->Reshape(num[0],channel[0],height[0],width[0]/2+1);
      Layer<float>::first_layer_weighted_sample_real[0].reset(new Blob<float>());
      Layer<float>::first_layer_weighted_sample_real[0]->Reshape(num[0],1,height[0],width[0]/2+1);
      Layer<float>::first_layer_weighted_sample_imag[0].reset(new Blob<float>());
      Layer<float>::first_layer_weighted_sample_imag[0]->Reshape(num[0],1,height[0],width[0]/2+1); 

      Layer<float>::first_layer_tmp_real1[0].reset(new Blob<float>());
      Layer<float>::first_layer_tmp_real1[0]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::first_layer_tmp_imag1[0].reset(new Blob<float>());
      Layer<float>::first_layer_tmp_imag1[0]->Reshape(1,channel[0],height[0],width[0]/2+1); 

       Layer<float>::H_total[0].reset(new Blob<float>());
       Layer<float>::H_total[0]->Reshape(frag_num[0],channel[0],height[0],width[0]);  

       Layer<float>::second_layer_hf_real[0].reset(new Blob<float>());
       Layer<float>::second_layer_hf_real[0]->Reshape(frag_num[0],channel[0],height[0],width[0]/2+1);

       Layer<float>::second_layer_hf_imag[0].reset(new Blob<float>());
       Layer<float>::second_layer_hf_imag[0]->Reshape(frag_num[0],channel[0],height[0],width[0]/2+1);

       Layer<float>::filter_H[0].reset(new Blob<float>());
       Layer<float>::filter_H[0]->Reshape(1,channel[0],height[0],width[0]);

       Layer<float>::second_layer_weighted_sample_real[0].reset(new Blob<float>());
       Layer<float>::second_layer_weighted_sample_real[0]->Reshape(num[0]*frag_num[0],1,height[0],width[0]/2+1);

       Layer<float>::second_layer_weighted_sample_imag[0].reset(new Blob<float>());
       Layer<float>::second_layer_weighted_sample_imag[0]->Reshape(num[0]*frag_num[0],1,height[0],width[0]/2+1);


    }
    else if(net_index[0]==2)
    {
      Layer<float>::input_xff_real[1].reset(new Blob<float>());
      Layer<float>::input_xff_real[1]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::input_xff_imag[1].reset(new Blob<float>());
      Layer<float>::input_xff_imag[1]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::input_yff_real[1].reset(new Blob<float>());
      Layer<float>::input_yff_real[1]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::input_yff_imag[1].reset(new Blob<float>());
      Layer<float>::input_yff_imag[1]->Reshape(1,channel[0],height[0],width[0]/2+1);  
       
      Layer<float>::hf1[1].reset(new Blob<float>());
      Layer<float>::hf1[1]->Reshape(1,channel[0],height[0],width[0]/2+1);  

      Layer<float>::first_layer_hf_real[1].reset(new Blob<float>());
      Layer<float>::first_layer_hf_real[1]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::first_layer_hf_imag[1].reset(new Blob<float>());
      Layer<float>::first_layer_hf_imag[1]->Reshape(1,channel[0],height[0],width[0]/2+1);

      Layer<float>::hf_tmp_real[1].reset(new Blob<float>());
      Layer<float>::hf_tmp_real[1]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::hf_tmp_imag[1].reset(new Blob<float>());
      Layer<float>::hf_tmp_imag[1]->Reshape(1,channel[0],height[0],width[0]/2+1);

      Layer<float>::matlab_hf_real[1].reset(new Blob<float>());
      Layer<float>::matlab_hf_real[1]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::matlab_hf_imag[1].reset(new Blob<float>());
      Layer<float>::matlab_hf_imag[1]->Reshape(1,channel[0],height[0],width[0]/2+1); 

      Layer<float>::first_layer_samplef_real[1].reset(new Blob<float>());
      Layer<float>::first_layer_samplef_real[1]->Reshape(num[0],channel[0],height[0],width[0]/2+1);
      Layer<float>::first_layer_samplef_imag[1].reset(new Blob<float>());
      Layer<float>::first_layer_samplef_imag[1]->Reshape(num[0],channel[0],height[0],width[0]/2+1); 
 
      Layer<float>::laplace_real[1].reset(new Blob<float>());
      Layer<float>::laplace_real[1]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::laplace_imag[1].reset(new Blob<float>());
      Layer<float>::laplace_imag[1]->Reshape(1,channel[0],height[0],width[0]/2+1); 
  

      Layer<float>::first_layer_weighted_sample_real[1].reset(new Blob<float>());
      Layer<float>::first_layer_weighted_sample_real[1]->Reshape(num[0],1,height[0],width[0]/2+1);
      Layer<float>::first_layer_weighted_sample_imag[1].reset(new Blob<float>());
      Layer<float>::first_layer_weighted_sample_imag[1]->Reshape(num[0],1,height[0],width[0]/2+1);  

      Layer<float>::first_layer_tmp_real1[1].reset(new Blob<float>());
      Layer<float>::first_layer_tmp_real1[1]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::first_layer_tmp_imag1[1].reset(new Blob<float>());
      Layer<float>::first_layer_tmp_imag1[1]->Reshape(1,channel[0],height[0],width[0]/2+1); 

      Layer<float>::KK_real[1].reset(new Blob<float>());
      Layer<float>::KK_imag[1].reset(new Blob<float>());
     
      Layer<float>::KK_real[1]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::KK_imag[1]->Reshape(1,channel[0],height[0],width[0]/2+1);
  
      Layer<float>::H_total[1].reset(new Blob<float>());
      Layer<float>::H_total[1]->Reshape(frag_num[0],channel[0],height[0],width[0]);  
  
      Layer<float>::filter_H[1].reset(new Blob<float>());
      Layer<float>::filter_H[1]->Reshape(1,channel[0],height[0],width[0]);

       Layer<float>::second_layer_hf_real[1].reset(new Blob<float>());
       Layer<float>::second_layer_hf_real[1]->Reshape(frag_num[0],channel[0],height[0],width[0]/2+1);

       Layer<float>::second_layer_hf_imag[1].reset(new Blob<float>());
       Layer<float>::second_layer_hf_imag[1]->Reshape(frag_num[0],channel[0],height[0],width[0]/2+1);

       Layer<float>::second_layer_weighted_sample_real[1].reset(new Blob<float>());
       Layer<float>::second_layer_weighted_sample_real[1]->Reshape(num[0]*frag_num[0],1,height[0],width[0]/2+1);

       Layer<float>::second_layer_weighted_sample_imag[1].reset(new Blob<float>());
       Layer<float>::second_layer_weighted_sample_imag[1]->Reshape(num[0]*frag_num[0],1,height[0],width[0]/2+1);

    }
    else if(net_index[0]==3)
    {
      Layer<float>::input_xff_real[2].reset(new Blob<float>());
      Layer<float>::input_xff_real[2]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::input_xff_imag[2].reset(new Blob<float>());
      Layer<float>::input_xff_imag[2]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::input_yff_real[2].reset(new Blob<float>());
      Layer<float>::input_yff_real[2]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::input_yff_imag[2].reset(new Blob<float>());
      Layer<float>::input_yff_imag[2]->Reshape(1,channel[0],height[0],width[0]/2+1);  

      Layer<float>::hf1[2].reset(new Blob<float>());
      Layer<float>::hf1[2]->Reshape(1,channel[0],height[0],width[0]/2+1); 

      Layer<float>::first_layer_hf_real[2].reset(new Blob<float>());
      Layer<float>::first_layer_hf_real[2]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::first_layer_hf_imag[2].reset(new Blob<float>());
      Layer<float>::first_layer_hf_imag[2]->Reshape(1,channel[0],height[0],width[0]/2+1);
 
      Layer<float>::hf_tmp_real[2].reset(new Blob<float>());
      Layer<float>::hf_tmp_real[2]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::hf_tmp_imag[2].reset(new Blob<float>());
      Layer<float>::hf_tmp_imag[2]->Reshape(1,channel[0],height[0],width[0]/2+1);

      Layer<float>::matlab_hf_real[2].reset(new Blob<float>());
      Layer<float>::matlab_hf_real[2]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::matlab_hf_imag[2].reset(new Blob<float>());
      Layer<float>::matlab_hf_imag[2]->Reshape(1,channel[0],height[0],width[0]/2+1); 

      Layer<float>::first_layer_samplef_real[2].reset(new Blob<float>());
      Layer<float>::first_layer_samplef_real[2]->Reshape(num[0],channel[0],height[0],width[0]/2+1);
      Layer<float>::first_layer_samplef_imag[2].reset(new Blob<float>());
      Layer<float>::first_layer_samplef_imag[2]->Reshape(num[0],channel[0],height[0],width[0]/2+1); 

      Layer<float>::laplace_real[2].reset(new Blob<float>());
      Layer<float>::laplace_real[2]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::laplace_imag[2].reset(new Blob<float>());
      Layer<float>::laplace_imag[2]->Reshape(1,channel[0],height[0],width[0]/2+1); 


      Layer<float>::first_layer_weighted_sample_real[2].reset(new Blob<float>());
      Layer<float>::first_layer_weighted_sample_real[2]->Reshape(num[0],1,height[0],width[0]/2+1);
      Layer<float>::first_layer_weighted_sample_imag[2].reset(new Blob<float>());
      Layer<float>::first_layer_weighted_sample_imag[2]->Reshape(num[0],1,height[0],width[0]/2+1); 

      Layer<float>::first_layer_tmp_real1[2].reset(new Blob<float>());
      Layer<float>::first_layer_tmp_real1[2]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::first_layer_tmp_imag1[2].reset(new Blob<float>());
      Layer<float>::first_layer_tmp_imag1[2]->Reshape(1,channel[0],height[0],width[0]/2+1);  

      Layer<float>::cropped_yf_real.resize(1);
      Layer<float>::cropped_yf_real[0].reset(new Blob<float>()); 
      Layer<float>::cropped_yf_real[0]->Reshape(num[0],1,height[0],width[0]/2+1);

      Layer<float>::cropped_yf_imag.resize(1);
      Layer<float>::cropped_yf_imag[0].reset(new Blob<float>()); 
      Layer<float>::cropped_yf_imag[0]->Reshape(num[0],1,height[0],width[0]/2+1);

      Layer<float>::KK_real[2].reset(new Blob<float>());
      Layer<float>::KK_imag[2].reset(new Blob<float>());
     
      Layer<float>::KK_real[2]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::KK_imag[2]->Reshape(1,channel[0],height[0],width[0]/2+1);

      Layer<float>::H_total[2].reset(new Blob<float>());
      Layer<float>::H_total[2]->Reshape(frag_num[0],channel[0],height[0],width[0]); 

      Layer<float>::filter_H[2].reset(new Blob<float>());
      Layer<float>::filter_H[2]->Reshape(1,channel[0],height[0],width[0]);  

       Layer<float>::second_layer_hf_real[2].reset(new Blob<float>());
       Layer<float>::second_layer_hf_real[2]->Reshape(frag_num[0],channel[0],height[0],width[0]/2+1);

       Layer<float>::second_layer_hf_imag[2].reset(new Blob<float>());
       Layer<float>::second_layer_hf_imag[2]->Reshape(frag_num[0],channel[0],height[0],width[0]/2+1);

       Layer<float>::second_layer_weighted_sample_real[2].reset(new Blob<float>());
       Layer<float>::second_layer_weighted_sample_real[2]->Reshape(num[0]*frag_num[0],1,height[0],width[0]/2+1);

       Layer<float>::second_layer_weighted_sample_imag[2].reset(new Blob<float>());
       Layer<float>::second_layer_weighted_sample_imag[2]->Reshape(num[0]*frag_num[0],1,height[0],width[0]/2+1);

    }
    else if(net_index[0]==4)
    {
      Layer<float>::input_xff_real[3].reset(new Blob<float>());
      Layer<float>::input_xff_real[3]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::input_xff_imag[3].reset(new Blob<float>());
      Layer<float>::input_xff_imag[3]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::input_yff_real[3].reset(new Blob<float>());
      Layer<float>::input_yff_real[3]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::input_yff_imag[3].reset(new Blob<float>());
      Layer<float>::input_yff_imag[3]->Reshape(1,channel[0],height[0],width[0]/2+1);  

      Layer<float>::hf1[3].reset(new Blob<float>());
      Layer<float>::hf1[3]->Reshape(1,channel[0],height[0],width[0]/2+1); 

      Layer<float>::first_layer_hf_real[3].reset(new Blob<float>());
      Layer<float>::first_layer_hf_real[3]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::first_layer_hf_imag[3].reset(new Blob<float>());
      Layer<float>::first_layer_hf_imag[3]->Reshape(1,channel[0],height[0],width[0]/2+1);

      Layer<float>::hf_tmp_real[3].reset(new Blob<float>());
      Layer<float>::hf_tmp_real[3]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::hf_tmp_imag[3].reset(new Blob<float>());
      Layer<float>::hf_tmp_imag[3]->Reshape(1,channel[0],height[0],width[0]/2+1);

      Layer<float>::matlab_hf_real[3].reset(new Blob<float>());
      Layer<float>::matlab_hf_real[3]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::matlab_hf_imag[3].reset(new Blob<float>());
      Layer<float>::matlab_hf_imag[3]->Reshape(1,channel[0],height[0],width[0]/2+1); 

      Layer<float>::first_layer_samplef_real[3].reset(new Blob<float>());
      Layer<float>::first_layer_samplef_real[3]->Reshape(num[0],channel[0],height[0],width[0]/2+1);
      Layer<float>::first_layer_samplef_imag[3].reset(new Blob<float>());
      Layer<float>::first_layer_samplef_imag[3]->Reshape(num[0],channel[0],height[0],width[0]/2+1); 

      Layer<float>::laplace_real[3].reset(new Blob<float>());
      Layer<float>::laplace_real[3]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::laplace_imag[3].reset(new Blob<float>());
      Layer<float>::laplace_imag[3]->Reshape(1,channel[0],height[0],width[0]/2+1); 


      Layer<float>::first_layer_weighted_sample_real[3].reset(new Blob<float>());
      Layer<float>::first_layer_weighted_sample_real[3]->Reshape(num[0],1,height[0],width[0]/2+1);
      Layer<float>::first_layer_weighted_sample_imag[3].reset(new Blob<float>());
      Layer<float>::first_layer_weighted_sample_imag[3]->Reshape(num[0],1,height[0],width[0]/2+1);

      Layer<float>::first_layer_tmp_real1[3].reset(new Blob<float>());
      Layer<float>::first_layer_tmp_real1[3]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::first_layer_tmp_imag1[3].reset(new Blob<float>());
      Layer<float>::first_layer_tmp_imag1[3]->Reshape(1,channel[0],height[0],width[0]/2+1);  

      Layer<float>::KK_real[3].reset(new Blob<float>());
      Layer<float>::KK_imag[3].reset(new Blob<float>());
     
      Layer<float>::KK_real[3]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::KK_imag[3]->Reshape(1,channel[0],height[0],width[0]/2+1);  

      Layer<float>::H_total[3].reset(new Blob<float>());
      Layer<float>::H_total[3]->Reshape(frag_num[0],channel[0],height[0],width[0]);  

      Layer<float>::filter_H[3].reset(new Blob<float>());
      Layer<float>::filter_H[3]->Reshape(1,channel[0],height[0],width[0]); 

       Layer<float>::second_layer_hf_real[3].reset(new Blob<float>());
       Layer<float>::second_layer_hf_real[3]->Reshape(frag_num[0],channel[0],height[0],width[0]/2+1);

       Layer<float>::second_layer_hf_imag[3].reset(new Blob<float>());
       Layer<float>::second_layer_hf_imag[3]->Reshape(frag_num[0],channel[0],height[0],width[0]/2+1);

       Layer<float>::second_layer_weighted_sample_real[3].reset(new Blob<float>());
       Layer<float>::second_layer_weighted_sample_real[3]->Reshape(num[0]*frag_num[0],1,height[0],width[0]/2+1);

       Layer<float>::second_layer_weighted_sample_imag[3].reset(new Blob<float>());
       Layer<float>::second_layer_weighted_sample_imag[3]->Reshape(num[0]*frag_num[0],1,height[0],width[0]/2+1);
    } 

    else if(net_index[0]==5)
    {
      Layer<float>::input_xff_real[4].reset(new Blob<float>());
      Layer<float>::input_xff_real[4]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::input_xff_imag[4].reset(new Blob<float>());
      Layer<float>::input_xff_imag[4]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::input_yff_real[4].reset(new Blob<float>());
      Layer<float>::input_yff_real[4]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::input_yff_imag[4].reset(new Blob<float>());
      Layer<float>::input_yff_imag[4]->Reshape(1,channel[0],height[0],width[0]/2+1);  

      Layer<float>::hf1[4].reset(new Blob<float>());
      Layer<float>::hf1[4]->Reshape(1,channel[0],height[0],width[0]/2+1); 

      Layer<float>::first_layer_hf_real[4].reset(new Blob<float>());
      Layer<float>::first_layer_hf_real[4]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::first_layer_hf_imag[4].reset(new Blob<float>());
      Layer<float>::first_layer_hf_imag[4]->Reshape(1,channel[0],height[0],width[0]/2+1);

      Layer<float>::hf_tmp_real[4].reset(new Blob<float>());
      Layer<float>::hf_tmp_real[4]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::hf_tmp_imag[4].reset(new Blob<float>());
      Layer<float>::hf_tmp_imag[4]->Reshape(1,channel[0],height[0],width[0]/2+1);

      Layer<float>::matlab_hf_real[4].reset(new Blob<float>());
      Layer<float>::matlab_hf_real[4]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::matlab_hf_imag[4].reset(new Blob<float>());
      Layer<float>::matlab_hf_imag[4]->Reshape(1,channel[0],height[0],width[0]/2+1); 

      Layer<float>::first_layer_samplef_real[4].reset(new Blob<float>());
      Layer<float>::first_layer_samplef_real[4]->Reshape(num[0],channel[0],height[0],width[0]/2+1);
      Layer<float>::first_layer_samplef_imag[4].reset(new Blob<float>());
      Layer<float>::first_layer_samplef_imag[4]->Reshape(num[0],channel[0],height[0],width[0]/2+1); 

      Layer<float>::laplace_real[4].reset(new Blob<float>());
      Layer<float>::laplace_real[4]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::laplace_imag[4].reset(new Blob<float>());
      Layer<float>::laplace_imag[4]->Reshape(1,channel[0],height[0],width[0]/2+1); 

      Layer<float>::first_layer_weighted_sample_real[4].reset(new Blob<float>());
      Layer<float>::first_layer_weighted_sample_real[4]->Reshape(num[0],1,height[0],width[0]/2+1);
      Layer<float>::first_layer_weighted_sample_imag[4].reset(new Blob<float>());
      Layer<float>::first_layer_weighted_sample_imag[4]->Reshape(num[0],1,height[0],width[0]/2+1); 

      Layer<float>::first_layer_tmp_real1[4].reset(new Blob<float>());
      Layer<float>::first_layer_tmp_real1[4]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::first_layer_tmp_imag1[4].reset(new Blob<float>());
      Layer<float>::first_layer_tmp_imag1[4]->Reshape(1,channel[0],height[0],width[0]/2+1);  

      Layer<float>::KK_real[4].reset(new Blob<float>());
      Layer<float>::KK_imag[4].reset(new Blob<float>());
     
      Layer<float>::KK_real[4]->Reshape(1,channel[0],height[0],width[0]/2+1);
      Layer<float>::KK_imag[4]->Reshape(1,channel[0],height[0],width[0]/2+1); 
  
      Layer<float>::H_total[4].reset(new Blob<float>());
      Layer<float>::H_total[4]->Reshape(frag_num[0],channel[0],height[0],width[0]);  

      Layer<float>::filter_H[4].reset(new Blob<float>());
      Layer<float>::filter_H[4]->Reshape(1,channel[0],height[0],width[0]); 

       Layer<float>::second_layer_hf_real[4].reset(new Blob<float>());
       Layer<float>::second_layer_hf_real[4]->Reshape(frag_num[0],channel[0],height[0],width[0]/2+1);

       Layer<float>::second_layer_hf_imag[4].reset(new Blob<float>());
       Layer<float>::second_layer_hf_imag[4]->Reshape(frag_num[0],channel[0],height[0],width[0]/2+1);

       Layer<float>::second_layer_weighted_sample_real[4].reset(new Blob<float>());
       Layer<float>::second_layer_weighted_sample_real[4]->Reshape(num[0]*frag_num[0],1,height[0],width[0]/2+1);

       Layer<float>::second_layer_weighted_sample_imag[4].reset(new Blob<float>());
       Layer<float>::second_layer_weighted_sample_imag[4]->Reshape(num[0]*frag_num[0],1,height[0],width[0]/2+1);

    }   
}

static void get_data_from_matlab(MEX_ARGS) {
   mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
  const int* dims = static_cast<const int*>(mxGetDimensions(prhs[1]));
    
  // 首先从matlab中输入hf的实部和虚部 
    //
  const  mxArray* mx_mat3=prhs[3];
  const float* net_index = reinterpret_cast<const float*>(mxGetData(mx_mat3));
  Blob<float>* tmp=new Blob<float>(); //存储fft的实部
  tmp->Reshape(1,dims[2],dims[1],dims[0]);
  Blob<float>* tmp1=new Blob<float>(); //存储fft的虚部
  tmp1->Reshape(1,dims[2],dims[1],dims[0]);
  mx_mat_to_blob(prhs[1],tmp, DATA);
  mx_mat_to_blob(prhs[2],tmp1, DATA);
 
   caffe_copy(tmp->count(),tmp->mutable_gpu_data(),Layer<float>::first_layer_hf_real[net_index[0]-1]->mutable_gpu_data()); 
   caffe_copy(tmp1->count(),tmp1->mutable_gpu_data(),Layer<float>::first_layer_hf_imag[net_index[0]-1]->mutable_gpu_data()); 
    
delete tmp;
delete tmp1;
}

static void set_frame_id(MEX_ARGS) {
   mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
    const  mxArray* mx_mat1=prhs[1];
    const float* frame_id = reinterpret_cast<const float*>(mxGetData(mx_mat1));
    
   Layer<float>::frame.resize(1);
   Layer<float>::frame[0].reset(new Blob<float>());
   Layer<float>::frame[0]->Reshape(1,1,1,1);
   float* frame_cpu=Layer<float>::frame[0]->mutable_cpu_data();
   frame_cpu[0]=frame_id[0];
}

static void update_samplesf(MEX_ARGS) {
   mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
  const int* dims = static_cast<const int*>(mxGetDimensions(prhs[1]));
    
  // 首先从matlab中输入hf的实部和虚部 
    //
  const  mxArray* mx_mat3=prhs[3];
  const float* net_index = reinterpret_cast<const float*>(mxGetData(mx_mat3));
  const  mxArray* mx_mat4=prhs[4];  
  const float* replace_id= reinterpret_cast<const float*>(mxGetData(mx_mat4)); 
  Blob<float>* tmp=new Blob<float>(); //存储fft的实部
  tmp->Reshape(1,dims[2],dims[1],dims[0]);
  Blob<float>* tmp1=new Blob<float>(); //存储fft的虚部
  tmp1->Reshape(1,dims[2],dims[1],dims[0]);
  mx_mat_to_blob(prhs[1],tmp, DATA);
  mx_mat_to_blob(prhs[2],tmp1, DATA);// 存储samplesf的实部及虚部

   caffe_copy(tmp->count(),tmp->mutable_gpu_data(),Layer<float>::first_layer_samplef_real[net_index[0]-1]->mutable_gpu_data()+Layer<float>::first_layer_samplef_real[net_index[0]-1]->offset(replace_id[0]-1)); 
  caffe_copy(tmp->count(),tmp1->mutable_gpu_data(),Layer<float>::first_layer_samplef_imag[net_index[0]-1]->mutable_gpu_data()+Layer<float>::first_layer_samplef_real[net_index[0]-1]->offset(replace_id[0]-1)); 

    
delete tmp;
delete tmp1;
}

static void get_samplesf(MEX_ARGS) {
   mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
    
  // 首先从matlab中输入hf的实部和虚部 
    //
  const  mxArray* mx_mat3=prhs[1];
  const float* net_index = reinterpret_cast<const float*>(mxGetData(mx_mat3)); 

  Blob<float>* tmp=new Blob<float>(); //存储fft的实部
  tmp->Reshape(Layer<float>::first_layer_samplef_real[net_index[0]-1]->num(),Layer<float>::first_layer_samplef_real[net_index[0]-1]->channels(),Layer<float>::first_layer_samplef_real[net_index[0]-1]->height(),Layer<float>::first_layer_samplef_real[net_index[0]-1]->width());
  Blob<float>* tmp1=new Blob<float>(); //存储fft的虚部
  tmp1->Reshape(Layer<float>::first_layer_samplef_real[net_index[0]-1]->num(),Layer<float>::first_layer_samplef_real[net_index[0]-1]->channels(),Layer<float>::first_layer_samplef_real[net_index[0]-1]->height(),Layer<float>::first_layer_samplef_real[net_index[0]-1]->width());

   caffe_copy(tmp->count(),Layer<float>::first_layer_samplef_real[net_index[0]-1]->mutable_gpu_data(),tmp->mutable_gpu_data()); 
  caffe_copy(tmp->count(),Layer<float>::first_layer_samplef_imag[net_index[0]-1]->mutable_gpu_data(),tmp1->mutable_gpu_data());
 
//printf("%d %d %d %d\n",tmp->height(),tmp->width(),tmp->channels(),tmp->num());

  plhs[0]=blob_to_mx_mat(tmp, DATA);
  plhs[1]=blob_to_mx_mat(tmp1, DATA);
    
delete tmp;
delete tmp1;
}


static void input_yf(MEX_ARGS) { 
  const int* dims = static_cast<const int*>(mxGetDimensions(prhs[1])); 
  
  Blob<float>* tmp=new Blob<float>(); //存储fft的实部
  tmp->Reshape(1,1,dims[1],dims[0]); 
  mx_mat_to_blob(prhs[1],tmp, DATA);
  Blob<float>* tmp1=new Blob<float>(); //存储fft的实部
  tmp1->Reshape(1,1,dims[1],dims[0]); 
  mx_mat_to_blob(prhs[2],tmp1, DATA);
  Blob<float>* tmp2=new Blob<float>(); //存储fft的实部
  tmp2->Reshape(1,1,dims[1],dims[0]); 
  mx_mat_to_blob(prhs[3],tmp2, DATA);    

  const  mxArray* mx_mat4=prhs[4];
  const float* num = reinterpret_cast<const float*>(mxGetData(mx_mat4));

  Layer<float>::first_layer_yf_real.resize(1);
  Layer<float>::first_layer_yf_real[0].reset(new Blob<float>());
  Layer<float>::first_layer_yf_real[0]->Reshape(1,1,dims[1],dims[0]);
  Layer<float>::first_layer_yf_imag.resize(1);
  Layer<float>::first_layer_yf_imag[0].reset(new Blob<float>());
  Layer<float>::first_layer_yf_imag[0]->Reshape(1,1,dims[1],dims[0]);

  Layer<float>::first_layer_yf_diff_real.resize(1);
  Layer<float>::first_layer_yf_diff_real[0].reset(new Blob<float>());
  Layer<float>::first_layer_yf_diff_real[0]->Reshape(num[0],1,dims[1],dims[0]);
  Layer<float>::first_layer_yf_diff_imag.resize(1);
  Layer<float>::first_layer_yf_diff_imag[0].reset(new Blob<float>());
  Layer<float>::first_layer_yf_diff_imag[0]->Reshape(num[0],1,dims[1],dims[0]); 

  Layer<float>::L_index.resize(1);
  Layer<float>::L_index[0].reset(new Blob<float>());
  Layer<float>::L_index[0]->Reshape(1,1,dims[1],dims[0]);


  caffe_copy(tmp->count(),tmp->mutable_gpu_data(),Layer<float>::first_layer_yf_real[0]->mutable_gpu_data());  
  caffe_copy(tmp1->count(),tmp1->mutable_gpu_data(),Layer<float>::first_layer_yf_imag[0]->mutable_gpu_data());  
  caffe_copy(tmp2->count(),tmp2->mutable_gpu_data(),Layer<float>::L_index[0]->mutable_gpu_data());  
 delete tmp;
 delete tmp1;
 delete tmp2;
}

static void set_L_index(MEX_ARGS) { 
  const int* dims = static_cast<const int*>(mxGetDimensions(prhs[1]));
  const int* dims1 = static_cast<const int*>(mxGetDimensions(prhs[2]));  
  
  Blob<float>* tmp=new Blob<float>(); //存储fft的实部
  tmp->Reshape(1,1,dims[1],dims[0]); 
  mx_mat_to_blob(prhs[1],tmp, DATA);

  Blob<float>* tmp1=new Blob<float>(); //存储fft的实部
  tmp1->Reshape(1,1,dims1[1],dims1[0]); 
  mx_mat_to_blob(prhs[2],tmp1, DATA);  

  Layer<float>::L_index.resize(1);
  Layer<float>::L_index[0].reset(new Blob<float>());
  Layer<float>::L_index[0]->Reshape(1,1,dims[1],dims[0]);

  Layer<float>::L_index1.resize(1);
  Layer<float>::L_index1[0].reset(new Blob<float>());
  Layer<float>::L_index1[0]->Reshape(1,1,dims1[1],dims1[0]);   
  caffe_copy(tmp->count(),tmp->mutable_gpu_data(),Layer<float>::L_index[0]->mutable_gpu_data());
  caffe_copy(tmp1->count(),tmp1->mutable_gpu_data(),Layer<float>::L_index1[0]->mutable_gpu_data());   
 delete tmp; delete tmp1;
}


static void input_sample_weight(MEX_ARGS) { 
  const int* dims = static_cast<const int*>(mxGetDimensions(prhs[1])); 
  
  Blob<float>* tmp=new Blob<float>(); //存储fft的实部
  tmp->Reshape(1,1,1,dims[0]); 
  mx_mat_to_blob(prhs[1],tmp, DATA);
  
  Layer<float>::sample_weight.resize(1);
  Layer<float>::sample_weight[0].reset(new Blob<float>());
  Layer<float>::sample_weight[0]->Reshape(1,1,1,dims[0]);

 caffe_copy(tmp->count(),tmp->mutable_gpu_data(),Layer<float>::sample_weight[0]->mutable_gpu_data());
 delete tmp;

}

static void set_reg_window(MEX_ARGS) { 
  const int* dims1 = static_cast<const int*>(mxGetDimensions(prhs[1]));
  const int* dims2 = static_cast<const int*>(mxGetDimensions(prhs[2]));    
  const int* dims3 = static_cast<const int*>(mxGetDimensions(prhs[3]));
  const int* dims4 = static_cast<const int*>(mxGetDimensions(prhs[4]));
  const int* dims5 = static_cast<const int*>(mxGetDimensions(prhs[5]));
  const int* dims6 = static_cast<const int*>(mxGetDimensions(prhs[6]));   

  Layer<float>::fftshift_mask.resize(2);
  Layer<float>::fftshift_mask[0].reset(new Blob<float>());
  Layer<float>::fftshift_mask[0]->Reshape(1,1,dims1[1],dims1[0]);
  Layer<float>::fftshift_mask[1].reset(new Blob<float>());
  Layer<float>::fftshift_mask[1]->Reshape(1,1,dims2[1],dims2[0]);

  Layer<float>::ifftshift_mask.resize(2);
  Layer<float>::ifftshift_mask[0].reset(new Blob<float>());
  Layer<float>::ifftshift_mask[0]->Reshape(1,1,dims3[1],dims3[0]);
  Layer<float>::ifftshift_mask[1].reset(new Blob<float>());
  Layer<float>::ifftshift_mask[1]->Reshape(1,1,dims4[1],dims4[0]); 

  Layer<float>::binary_mask.resize(2);
  Layer<float>::binary_mask[0].reset(new Blob<float>());
  Layer<float>::binary_mask[0]->Reshape(1,1,dims5[1],dims5[0]);
  Layer<float>::binary_mask[1].reset(new Blob<float>());
  Layer<float>::binary_mask[1]->Reshape(1,1,dims6[1],dims6[0]); 


  Blob<float>* tmp1=new Blob<float>();
  Blob<float>* tmp2=new Blob<float>();
  Blob<float>* tmp3=new Blob<float>();
  Blob<float>* tmp4=new Blob<float>();
  Blob<float>* tmp5=new Blob<float>();
  Blob<float>* tmp6=new Blob<float>();
  tmp1->Reshape(1,1,dims1[1],dims1[0]);
  tmp2->Reshape(1,1,dims2[1],dims2[0]); 
  tmp3->Reshape(1,1,dims3[1],dims3[0]);
  tmp4->Reshape(1,1,dims4[1],dims4[0]);  
  tmp5->Reshape(1,1,dims5[1],dims5[0]);
  tmp6->Reshape(1,1,dims6[1],dims6[0]);     
  mx_mat_to_blob(prhs[1],tmp1,DATA);
  mx_mat_to_blob(prhs[2],tmp2,DATA);
  mx_mat_to_blob(prhs[3],tmp3,DATA);
  mx_mat_to_blob(prhs[4],tmp4,DATA);
  mx_mat_to_blob(prhs[5],tmp5,DATA);
  mx_mat_to_blob(prhs[6],tmp6,DATA);
  caffe_copy(tmp1->count(),tmp1->mutable_gpu_data(),Layer<float>::fftshift_mask[0]->mutable_gpu_data()); 
  caffe_copy(tmp2->count(),tmp2->mutable_gpu_data(),Layer<float>::fftshift_mask[1]->mutable_gpu_data());
  caffe_copy(tmp3->count(),tmp3->mutable_gpu_data(),Layer<float>::ifftshift_mask[0]->mutable_gpu_data()); 
  caffe_copy(tmp4->count(),tmp4->mutable_gpu_data(),Layer<float>::ifftshift_mask[1]->mutable_gpu_data()); 
  caffe_copy(tmp5->count(),tmp5->mutable_gpu_data(),Layer<float>::binary_mask[0]->mutable_gpu_data()); 
  caffe_copy(tmp6->count(),tmp6->mutable_gpu_data(),Layer<float>::binary_mask[1]->mutable_gpu_data()); 

  delete tmp1; delete tmp2; delete tmp3; delete tmp4; delete tmp5; delete tmp6;

}

static void set_binary_mask_adaptive(MEX_ARGS) { 
  const int* dims1 = static_cast<const int*>(mxGetDimensions(prhs[1]));
  const int* dims2 = static_cast<const int*>(mxGetDimensions(prhs[2]));    

  Layer<float>::binary_mask_adaptive.resize(2);
  Layer<float>::binary_mask_adaptive[0].reset(new Blob<float>());
  Layer<float>::binary_mask_adaptive[0]->Reshape(1,1,dims1[1],dims1[0]);
  Layer<float>::binary_mask_adaptive[1].reset(new Blob<float>());
  Layer<float>::binary_mask_adaptive[1]->Reshape(1,1,dims2[1],dims2[0]); 


  Blob<float>* tmp1=new Blob<float>();
  Blob<float>* tmp2=new Blob<float>();
  tmp1->Reshape(1,1,dims1[1],dims1[0]);
  tmp2->Reshape(1,1,dims2[1],dims2[0]);      
  mx_mat_to_blob(prhs[1],tmp1,DATA);
  mx_mat_to_blob(prhs[2],tmp2,DATA);
  caffe_copy(tmp1->count(),tmp1->mutable_gpu_data(),Layer<float>::binary_mask_adaptive[0]->mutable_gpu_data()); 
  caffe_copy(tmp2->count(),tmp2->mutable_gpu_data(),Layer<float>::binary_mask_adaptive[1]->mutable_gpu_data());

  delete tmp1; delete tmp2;

}

static void set_index(MEX_ARGS) { 
  const int* dims1 = static_cast<const int*>(mxGetDimensions(prhs[1]));
  const  mxArray* mx_mat3=prhs[3];
  const float* total_num = reinterpret_cast<const float*>(mxGetData(mx_mat3));
  const  mxArray* mx_mat4=prhs[4];
  const float* total_num1 = reinterpret_cast<const float*>(mxGetData(mx_mat4)); 
  const  mxArray* mx_mat5=prhs[5];
  const float* total_num2 = reinterpret_cast<const float*>(mxGetData(mx_mat5)); 
  const  mxArray* mx_mat6=prhs[6];
  const float* total_num3 = reinterpret_cast<const float*>(mxGetData(mx_mat6));     

  Layer<float>::total_num.resize(1);
  Layer<float>::total_num[0].reset(new Blob<float>());
  Layer<float>::total_num[0]->Reshape(1,1,1,1);

  Layer<float>::total_num1.resize(1);
  Layer<float>::total_num1[0].reset(new Blob<float>());
  Layer<float>::total_num1[0]->Reshape(1,1,1,1);

  Layer<float>::total_num2.resize(1);
  Layer<float>::total_num2[0].reset(new Blob<float>());
  Layer<float>::total_num2[0]->Reshape(1,1,1,1);

  Layer<float>::total_num3.resize(1);
  Layer<float>::total_num3[0].reset(new Blob<float>());
  Layer<float>::total_num3[0]->Reshape(1,1,1,1);  


  float* total_num_cpu=Layer<float>::total_num[0]->mutable_cpu_data();
  total_num_cpu[0]=total_num[0];

  float* total_num_cpu1=Layer<float>::total_num1[0]->mutable_cpu_data();
  total_num_cpu1[0]=total_num1[0];

  float* total_num_cpu2=Layer<float>::total_num2[0]->mutable_cpu_data();
  total_num_cpu2[0]=total_num2[0];

  float* total_num_cpu3=Layer<float>::total_num3[0]->mutable_cpu_data();
  total_num_cpu3[0]=total_num3[0];


  Layer<float>::index.resize(1);
  Layer<float>::index[0].reset(new Blob<float>());
  Layer<float>::index[0]->Reshape(1,1,1,dims1[0]);

  Layer<float>::index1.resize(1);
  Layer<float>::index1[0].reset(new Blob<float>());
  Layer<float>::index1[0]->Reshape(1,1,1,dims1[0]);  

  Blob<float>* tmp=new Blob<float>();
  tmp->Reshape(1,1,1,dims1[0]);

  Blob<float>* tmp1=new Blob<float>();
  tmp1->Reshape(1,1,1,dims1[0]);  
 
  mx_mat_to_blob(prhs[1],tmp,DATA);
  mx_mat_to_blob(prhs[2],tmp1,DATA);

  caffe_copy(tmp->count(),tmp->mutable_gpu_data(),Layer<float>::index[0]->mutable_gpu_data()); 
  caffe_copy(tmp1->count(),tmp1->mutable_gpu_data(),Layer<float>::index1[0]->mutable_gpu_data());  
    
  delete tmp; delete tmp1;

}

static void clear_memory_function(MEX_ARGS) { 
  const  mxArray* mx_mat1=prhs[1];
  const float* clear_memory = reinterpret_cast<const float*>(mxGetData(mx_mat1));     

  float* clear_memory_cpu=Layer<float>::clear_memory[0]->mutable_cpu_data();
  clear_memory_cpu[0]=clear_memory[0];

}

static void set_hf_4(MEX_ARGS) {
   mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
  const int* dims1 = static_cast<const int*>(mxGetDimensions(prhs[1]));
  const int* dims2 = static_cast<const int*>(mxGetDimensions(prhs[3]));
  const int* dims3 = static_cast<const int*>(mxGetDimensions(prhs[5])); 
  const int* dims4 = static_cast<const int*>(mxGetDimensions(prhs[7]));   

  float* a=Layer<float>::feature_num[0]->mutable_cpu_data();
  a[0]=4; //定义4种feature


  Blob<float>* tmp1_real=new Blob<float>();
  tmp1_real->Reshape(1,dims1[2],dims1[1],dims1[0]);
  Blob<float>* tmp1_imag=new Blob<float>();
  tmp1_imag->Reshape(1,dims1[2],dims1[1],dims1[0]); 
  
  Blob<float>* tmp2_real=new Blob<float>();
  tmp2_real->Reshape(1,dims2[2],dims2[1],dims2[0]);
  Blob<float>* tmp2_imag=new Blob<float>();
  tmp2_imag->Reshape(1,dims2[2],dims2[1],dims2[0]);   

  Blob<float>* tmp3_real=new Blob<float>();
  tmp3_real->Reshape(1,dims3[2],dims3[1],dims3[0]);
  Blob<float>* tmp3_imag=new Blob<float>();
  tmp3_imag->Reshape(1,dims3[2],dims3[1],dims3[0]); 

  Blob<float>* tmp4_real=new Blob<float>();
  tmp4_real->Reshape(1,dims4[2],dims4[1],dims4[0]);
  Blob<float>* tmp4_imag=new Blob<float>();
  tmp4_imag->Reshape(1,dims4[2],dims4[1],dims4[0]);

  mx_mat_to_blob(prhs[1],tmp1_real,DATA); mx_mat_to_blob(prhs[2],tmp1_imag,DATA);
  mx_mat_to_blob(prhs[3],tmp2_real,DATA); mx_mat_to_blob(prhs[4],tmp2_imag,DATA);
  mx_mat_to_blob(prhs[5],tmp3_real,DATA); mx_mat_to_blob(prhs[6],tmp3_imag,DATA);
  mx_mat_to_blob(prhs[7],tmp4_real,DATA); mx_mat_to_blob(prhs[8],tmp4_imag,DATA);

  caffe_copy(Layer<float>::matlab_hf_real[0]->count(),tmp1_real->mutable_gpu_data(),Layer<float>::matlab_hf_real[0]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[0]->count(),tmp1_imag->mutable_gpu_data(),Layer<float>::matlab_hf_imag[0]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[1]->count(),tmp2_real->mutable_gpu_data(),Layer<float>::matlab_hf_real[1]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[1]->count(),tmp2_imag->mutable_gpu_data(),Layer<float>::matlab_hf_imag[1]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[2]->count(),tmp3_real->mutable_gpu_data(),Layer<float>::matlab_hf_real[2]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[2]->count(),tmp3_imag->mutable_gpu_data(),Layer<float>::matlab_hf_imag[2]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[3]->count(),tmp4_real->mutable_gpu_data(),Layer<float>::matlab_hf_real[3]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[3]->count(),tmp4_imag->mutable_gpu_data(),Layer<float>::matlab_hf_imag[3]->mutable_gpu_data());

  delete tmp1_real; delete tmp1_imag; delete tmp2_real; delete tmp2_imag; delete tmp3_real; delete tmp3_imag;
  delete tmp4_real; delete tmp4_imag;  

}

static void set_hf_5(MEX_ARGS) {
   mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
  const int* dims1 = static_cast<const int*>(mxGetDimensions(prhs[1]));
  const int* dims2 = static_cast<const int*>(mxGetDimensions(prhs[3]));
  const int* dims3 = static_cast<const int*>(mxGetDimensions(prhs[5])); 
  const int* dims4 = static_cast<const int*>(mxGetDimensions(prhs[7])); 
  const int* dims5 = static_cast<const int*>(mxGetDimensions(prhs[9]));   

  float* a=Layer<float>::feature_num[0]->mutable_cpu_data();
  a[0]=5; //定义5种feature


  Blob<float>* tmp1_real=new Blob<float>();
  tmp1_real->Reshape(1,dims1[2],dims1[1],dims1[0]);
  Blob<float>* tmp1_imag=new Blob<float>();
  tmp1_imag->Reshape(1,dims1[2],dims1[1],dims1[0]); 

  Blob<float>* tmp2_real=new Blob<float>();
  tmp2_real->Reshape(1,dims2[2],dims2[1],dims2[0]);
  Blob<float>* tmp2_imag=new Blob<float>();
  tmp2_imag->Reshape(1,dims2[2],dims2[1],dims2[0]);   

  Blob<float>* tmp3_real=new Blob<float>();
  tmp3_real->Reshape(1,dims3[2],dims3[1],dims3[0]);
  Blob<float>* tmp3_imag=new Blob<float>();
  tmp3_imag->Reshape(1,dims3[2],dims3[1],dims3[0]); 

  Blob<float>* tmp4_real=new Blob<float>();
  tmp4_real->Reshape(1,dims4[2],dims4[1],dims4[0]);
  Blob<float>* tmp4_imag=new Blob<float>();
  tmp4_imag->Reshape(1,dims4[2],dims4[1],dims4[0]);

  Blob<float>* tmp5_real=new Blob<float>();
  tmp5_real->Reshape(1,dims5[2],dims5[1],dims5[0]);
  Blob<float>* tmp5_imag=new Blob<float>();
  tmp5_imag->Reshape(1,dims5[2],dims5[1],dims5[0]);  

  mx_mat_to_blob(prhs[1],tmp1_real,DATA); mx_mat_to_blob(prhs[2],tmp1_imag,DATA);
  mx_mat_to_blob(prhs[3],tmp2_real,DATA); mx_mat_to_blob(prhs[4],tmp2_imag,DATA);
  mx_mat_to_blob(prhs[5],tmp3_real,DATA); mx_mat_to_blob(prhs[6],tmp3_imag,DATA);
  mx_mat_to_blob(prhs[7],tmp4_real,DATA); mx_mat_to_blob(prhs[8],tmp4_imag,DATA);
  mx_mat_to_blob(prhs[9],tmp5_real,DATA); mx_mat_to_blob(prhs[10],tmp5_imag,DATA);


  caffe_copy(Layer<float>::matlab_hf_real[0]->count(),tmp1_real->mutable_gpu_data(),Layer<float>::matlab_hf_real[0]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[0]->count(),tmp1_imag->mutable_gpu_data(),Layer<float>::matlab_hf_imag[0]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[1]->count(),tmp2_real->mutable_gpu_data(),Layer<float>::matlab_hf_real[1]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[1]->count(),tmp2_imag->mutable_gpu_data(),Layer<float>::matlab_hf_imag[1]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[2]->count(),tmp3_real->mutable_gpu_data(),Layer<float>::matlab_hf_real[2]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[2]->count(),tmp3_imag->mutable_gpu_data(),Layer<float>::matlab_hf_imag[2]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[3]->count(),tmp4_real->mutable_gpu_data(),Layer<float>::matlab_hf_real[3]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[3]->count(),tmp4_imag->mutable_gpu_data(),Layer<float>::matlab_hf_imag[3]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[4]->count(),tmp5_real->mutable_gpu_data(),Layer<float>::matlab_hf_real[4]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[4]->count(),tmp5_imag->mutable_gpu_data(),Layer<float>::matlab_hf_imag[4]->mutable_gpu_data());

  delete tmp1_real; delete tmp1_imag; delete tmp2_real; delete tmp2_imag; delete tmp3_real; delete tmp3_imag;
  delete tmp4_real; delete tmp4_imag; delete tmp5_real; delete tmp5_imag;

}

static void set_xf_yf4(MEX_ARGS) {
   mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
  const int* dims1 = static_cast<const int*>(mxGetDimensions(prhs[1]));
  const int* dims2 = static_cast<const int*>(mxGetDimensions(prhs[3]));
  const int* dims3 = static_cast<const int*>(mxGetDimensions(prhs[5])); 
  const int* dims4 = static_cast<const int*>(mxGetDimensions(prhs[7]));


  Blob<float>* tmp1_real=new Blob<float>();
  tmp1_real->Reshape(1,dims1[2],dims1[1],dims1[0]);
  Blob<float>* tmp1_imag=new Blob<float>();
  tmp1_imag->Reshape(1,dims1[2],dims1[1],dims1[0]); 
  
  Blob<float>* tmp2_real=new Blob<float>();
  tmp2_real->Reshape(1,dims2[2],dims2[1],dims2[0]);
  Blob<float>* tmp2_imag=new Blob<float>();
  tmp2_imag->Reshape(1,dims2[2],dims2[1],dims2[0]);   

  Blob<float>* tmp3_real=new Blob<float>();
  tmp3_real->Reshape(1,dims3[2],dims3[1],dims3[0]);
  Blob<float>* tmp3_imag=new Blob<float>();
  tmp3_imag->Reshape(1,dims3[2],dims3[1],dims3[0]); 

  Blob<float>* tmp4_real=new Blob<float>();
  tmp4_real->Reshape(1,dims4[2],dims4[1],dims4[0]);
  Blob<float>* tmp4_imag=new Blob<float>();
  tmp4_imag->Reshape(1,dims4[2],dims4[1],dims4[0]);

  mx_mat_to_blob(prhs[1],tmp1_real,DATA); mx_mat_to_blob(prhs[2],tmp1_imag,DATA);
  mx_mat_to_blob(prhs[3],tmp2_real,DATA); mx_mat_to_blob(prhs[4],tmp2_imag,DATA);
  mx_mat_to_blob(prhs[5],tmp3_real,DATA); mx_mat_to_blob(prhs[6],tmp3_imag,DATA);
  mx_mat_to_blob(prhs[7],tmp4_real,DATA); mx_mat_to_blob(prhs[8],tmp4_imag,DATA);

  caffe_copy(Layer<float>::matlab_hf_real[0]->count(),tmp1_real->mutable_gpu_data(),Layer<float>::input_xff_real[0]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[0]->count(),tmp1_imag->mutable_gpu_data(),Layer<float>::input_xff_imag[0]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[1]->count(),tmp2_real->mutable_gpu_data(),Layer<float>::input_xff_real[1]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[1]->count(),tmp2_imag->mutable_gpu_data(),Layer<float>::input_xff_imag[1]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[2]->count(),tmp3_real->mutable_gpu_data(),Layer<float>::input_xff_real[2]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[2]->count(),tmp3_imag->mutable_gpu_data(),Layer<float>::input_xff_imag[2]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[3]->count(),tmp4_real->mutable_gpu_data(),Layer<float>::input_xff_real[3]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[3]->count(),tmp4_imag->mutable_gpu_data(),Layer<float>::input_xff_imag[3]->mutable_gpu_data());

  mx_mat_to_blob(prhs[1+8],tmp1_real,DATA); mx_mat_to_blob(prhs[2+8],tmp1_imag,DATA);
  mx_mat_to_blob(prhs[3+8],tmp2_real,DATA); mx_mat_to_blob(prhs[4+8],tmp2_imag,DATA);
  mx_mat_to_blob(prhs[5+8],tmp3_real,DATA); mx_mat_to_blob(prhs[6+8],tmp3_imag,DATA);
  mx_mat_to_blob(prhs[7+8],tmp4_real,DATA); mx_mat_to_blob(prhs[8+8],tmp4_imag,DATA);

  caffe_copy(Layer<float>::matlab_hf_real[0]->count(),tmp1_real->mutable_gpu_data(),Layer<float>::input_yff_real[0]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[0]->count(),tmp1_imag->mutable_gpu_data(),Layer<float>::input_yff_imag[0]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[1]->count(),tmp2_real->mutable_gpu_data(),Layer<float>::input_yff_real[1]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[1]->count(),tmp2_imag->mutable_gpu_data(),Layer<float>::input_yff_imag[1]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[2]->count(),tmp3_real->mutable_gpu_data(),Layer<float>::input_yff_real[2]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[2]->count(),tmp3_imag->mutable_gpu_data(),Layer<float>::input_yff_imag[2]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[3]->count(),tmp4_real->mutable_gpu_data(),Layer<float>::input_yff_real[3]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[3]->count(),tmp4_imag->mutable_gpu_data(),Layer<float>::input_yff_imag[3]->mutable_gpu_data());

  delete tmp1_real; delete tmp1_imag; delete tmp2_real; delete tmp2_imag; delete tmp3_real; delete tmp3_imag;
  delete tmp4_real; delete tmp4_imag;  

}

static void set_xf_yf5(MEX_ARGS) {
   mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
  const int* dims1 = static_cast<const int*>(mxGetDimensions(prhs[1]));
  const int* dims2 = static_cast<const int*>(mxGetDimensions(prhs[3]));
  const int* dims3 = static_cast<const int*>(mxGetDimensions(prhs[5])); 
  const int* dims4 = static_cast<const int*>(mxGetDimensions(prhs[7]));
  const int* dims5 = static_cast<const int*>(mxGetDimensions(prhs[9]));   

  Layer<float>::inner_product.resize(1);
  Layer<float>::inner_product[0].reset(new Blob<float>());
  Layer<float>::inner_product[0]->Reshape(1,1,1,1);

  Blob<float>* tmp1_real=new Blob<float>();
  tmp1_real->Reshape(1,dims1[2],dims1[1],dims1[0]);
  Blob<float>* tmp1_imag=new Blob<float>();
  tmp1_imag->Reshape(1,dims1[2],dims1[1],dims1[0]); 
  
  Blob<float>* tmp2_real=new Blob<float>();
  tmp2_real->Reshape(1,dims2[2],dims2[1],dims2[0]);
  Blob<float>* tmp2_imag=new Blob<float>();
  tmp2_imag->Reshape(1,dims2[2],dims2[1],dims2[0]);   

  Blob<float>* tmp3_real=new Blob<float>();
  tmp3_real->Reshape(1,dims3[2],dims3[1],dims3[0]);
  Blob<float>* tmp3_imag=new Blob<float>();
  tmp3_imag->Reshape(1,dims3[2],dims3[1],dims3[0]); 

  Blob<float>* tmp4_real=new Blob<float>();
  tmp4_real->Reshape(1,dims4[2],dims4[1],dims4[0]);
  Blob<float>* tmp4_imag=new Blob<float>();
  tmp4_imag->Reshape(1,dims4[2],dims4[1],dims4[0]);

  Blob<float>* tmp5_real=new Blob<float>();
  tmp5_real->Reshape(1,dims5[2],dims5[1],dims5[0]);
  Blob<float>* tmp5_imag=new Blob<float>();
  tmp5_imag->Reshape(1,dims5[2],dims5[1],dims5[0]);


  mx_mat_to_blob(prhs[1],tmp1_real,DATA); mx_mat_to_blob(prhs[2],tmp1_imag,DATA);
  mx_mat_to_blob(prhs[3],tmp2_real,DATA); mx_mat_to_blob(prhs[4],tmp2_imag,DATA);
  mx_mat_to_blob(prhs[5],tmp3_real,DATA); mx_mat_to_blob(prhs[6],tmp3_imag,DATA);
  mx_mat_to_blob(prhs[7],tmp4_real,DATA); mx_mat_to_blob(prhs[8],tmp4_imag,DATA);
  mx_mat_to_blob(prhs[9],tmp5_real,DATA); mx_mat_to_blob(prhs[10],tmp5_imag,DATA);  

  caffe_copy(Layer<float>::matlab_hf_real[0]->count(),tmp1_real->mutable_gpu_data(),Layer<float>::input_xff_real[0]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[0]->count(),tmp1_imag->mutable_gpu_data(),Layer<float>::input_xff_imag[0]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[1]->count(),tmp2_real->mutable_gpu_data(),Layer<float>::input_xff_real[1]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[1]->count(),tmp2_imag->mutable_gpu_data(),Layer<float>::input_xff_imag[1]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[2]->count(),tmp3_real->mutable_gpu_data(),Layer<float>::input_xff_real[2]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[2]->count(),tmp3_imag->mutable_gpu_data(),Layer<float>::input_xff_imag[2]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[3]->count(),tmp4_real->mutable_gpu_data(),Layer<float>::input_xff_real[3]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[3]->count(),tmp4_imag->mutable_gpu_data(),Layer<float>::input_xff_imag[3]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_real[4]->count(),tmp5_real->mutable_gpu_data(),Layer<float>::input_xff_real[4]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[4]->count(),tmp5_imag->mutable_gpu_data(),Layer<float>::input_xff_imag[4]->mutable_gpu_data());

  mx_mat_to_blob(prhs[1+10],tmp1_real,DATA); mx_mat_to_blob(prhs[2+10],tmp1_imag,DATA);
  mx_mat_to_blob(prhs[3+10],tmp2_real,DATA); mx_mat_to_blob(prhs[4+10],tmp2_imag,DATA);
  mx_mat_to_blob(prhs[5+10],tmp3_real,DATA); mx_mat_to_blob(prhs[6+10],tmp3_imag,DATA);
  mx_mat_to_blob(prhs[7+10],tmp4_real,DATA); mx_mat_to_blob(prhs[8+10],tmp4_imag,DATA);
  mx_mat_to_blob(prhs[9+10],tmp5_real,DATA); mx_mat_to_blob(prhs[10+10],tmp5_imag,DATA);

  caffe_copy(Layer<float>::matlab_hf_real[0]->count(),tmp1_real->mutable_gpu_data(),Layer<float>::input_yff_real[0]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[0]->count(),tmp1_imag->mutable_gpu_data(),Layer<float>::input_yff_imag[0]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[1]->count(),tmp2_real->mutable_gpu_data(),Layer<float>::input_yff_real[1]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[1]->count(),tmp2_imag->mutable_gpu_data(),Layer<float>::input_yff_imag[1]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[2]->count(),tmp3_real->mutable_gpu_data(),Layer<float>::input_yff_real[2]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[2]->count(),tmp3_imag->mutable_gpu_data(),Layer<float>::input_yff_imag[2]->mutable_gpu_data()); 
  caffe_copy(Layer<float>::matlab_hf_real[3]->count(),tmp4_real->mutable_gpu_data(),Layer<float>::input_yff_real[3]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[3]->count(),tmp4_imag->mutable_gpu_data(),Layer<float>::input_yff_imag[3]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_real[4]->count(),tmp5_real->mutable_gpu_data(),Layer<float>::input_yff_real[4]->mutable_gpu_data());
  caffe_copy(Layer<float>::matlab_hf_imag[4]->count(),tmp5_imag->mutable_gpu_data(),Layer<float>::input_yff_imag[4]->mutable_gpu_data());

  delete tmp1_real; delete tmp1_imag; delete tmp2_real; delete tmp2_imag; delete tmp3_real; delete tmp3_imag;
  delete tmp4_real; delete tmp4_imag; delete tmp5_real; delete tmp5_imag; 

}


static void set_H_4(MEX_ARGS) {

  const int* dims1 = static_cast<const int*>(mxGetDimensions(prhs[1]));
  const int* dims2 = static_cast<const int*>(mxGetDimensions(prhs[2]));
  const int* dims3 = static_cast<const int*>(mxGetDimensions(prhs[3])); 
  const int* dims4 = static_cast<const int*>(mxGetDimensions(prhs[4]));  

    const  mxArray* mx_mat1=prhs[5];
    const float*  frag_num= reinterpret_cast<const float*>(mxGetData(mx_mat1));

  Blob<float>* tmp1=new Blob<float>();
  tmp1->Reshape(1,dims1[2],dims1[1],dims1[0]);
  Blob<float>* tmp2=new Blob<float>();
  tmp2->Reshape(1,dims2[2],dims2[1],dims2[0]); 
  Blob<float>* tmp3=new Blob<float>();
  tmp3->Reshape(1,dims3[2],dims3[1],dims3[0]);
  Blob<float>* tmp4=new Blob<float>();
  tmp4->Reshape(1,dims4[2],dims4[1],dims4[0]);    
  mx_mat_to_blob(prhs[1],tmp1,DATA); mx_mat_to_blob(prhs[2],tmp2,DATA); mx_mat_to_blob(prhs[3],tmp3,DATA);
  mx_mat_to_blob(prhs[4],tmp4,DATA); 
  caffe_copy(tmp1->count(),tmp1->mutable_gpu_data(),Layer<float>::filter_H[0]->mutable_gpu_data());
  caffe_copy(tmp2->count(),tmp2->mutable_gpu_data(),Layer<float>::filter_H[1]->mutable_gpu_data());
  caffe_copy(tmp3->count(),tmp3->mutable_gpu_data(),Layer<float>::filter_H[2]->mutable_gpu_data()); 
  caffe_copy(tmp4->count(),tmp4->mutable_gpu_data(),Layer<float>::filter_H[3]->mutable_gpu_data()); 

  delete tmp1; delete tmp2; delete tmp3; delete tmp4;


}

static void set_H_5(MEX_ARGS) {

  const int* dims1 = static_cast<const int*>(mxGetDimensions(prhs[1]));
  const int* dims2 = static_cast<const int*>(mxGetDimensions(prhs[2]));
  const int* dims3 = static_cast<const int*>(mxGetDimensions(prhs[3])); 
  const int* dims4 = static_cast<const int*>(mxGetDimensions(prhs[4])); 
  const int* dims5 = static_cast<const int*>(mxGetDimensions(prhs[5]));   

    const  mxArray* mx_mat1=prhs[6];
    const float*  frag_num= reinterpret_cast<const float*>(mxGetData(mx_mat1));

  Blob<float>* tmp1=new Blob<float>();
  tmp1->Reshape(1,dims1[2],dims1[1],dims1[0]);
  Blob<float>* tmp2=new Blob<float>();
  tmp2->Reshape(1,dims2[2],dims2[1],dims2[0]); 
  Blob<float>* tmp3=new Blob<float>();
  tmp3->Reshape(1,dims3[2],dims3[1],dims3[0]);
  Blob<float>* tmp4=new Blob<float>();
  tmp4->Reshape(1,dims4[2],dims4[1],dims4[0]);
  Blob<float>* tmp5=new Blob<float>();
  tmp5->Reshape(1,dims5[2],dims5[1],dims5[0]);    
  mx_mat_to_blob(prhs[1],tmp1,DATA); mx_mat_to_blob(prhs[2],tmp2,DATA); mx_mat_to_blob(prhs[3],tmp3,DATA);
  mx_mat_to_blob(prhs[4],tmp4,DATA); mx_mat_to_blob(prhs[5],tmp5,DATA);
  caffe_copy(tmp1->count(),tmp1->mutable_gpu_data(),Layer<float>::filter_H[0]->mutable_gpu_data());
  caffe_copy(tmp2->count(),tmp2->mutable_gpu_data(),Layer<float>::filter_H[1]->mutable_gpu_data());
  caffe_copy(tmp3->count(),tmp3->mutable_gpu_data(),Layer<float>::filter_H[2]->mutable_gpu_data()); 
  caffe_copy(tmp4->count(),tmp4->mutable_gpu_data(),Layer<float>::filter_H[3]->mutable_gpu_data());    
  caffe_copy(tmp5->count(),tmp5->mutable_gpu_data(),Layer<float>::filter_H[4]->mutable_gpu_data()); 

  delete tmp1; delete tmp2; delete tmp3; delete tmp4; delete tmp5;


}


static void set_patch_mask(MEX_ARGS) {
   mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
  const int* dims1 = static_cast<const int*>(mxGetDimensions(prhs[1]));
  const int* dims2 = static_cast<const int*>(mxGetDimensions(prhs[2]));

  Layer<float>::patch_mask.resize(2);
  Layer<float>::patch_mask[0].reset(new Blob<float>());
  Layer<float>::patch_mask[0]->Reshape(1,dims1[2],dims1[1],dims1[0]);
  Layer<float>::patch_mask[1].reset(new Blob<float>());
  Layer<float>::patch_mask[1]->Reshape(1,dims2[2],dims2[1],dims2[0]);

  Blob<float>* tmp1=new Blob<float>();
  tmp1->Reshape(1,dims1[2],dims1[1],dims1[0]);
  Blob<float>* tmp2=new Blob<float>();
  tmp2->Reshape(1,dims2[2],dims2[1],dims2[0]);    
  
  mx_mat_to_blob(prhs[1],tmp1,DATA);
  mx_mat_to_blob(prhs[2],tmp2,DATA);

  caffe_copy(tmp1->count(),tmp1->mutable_gpu_data(),Layer<float>::patch_mask[0]->mutable_gpu_data());
  caffe_copy(tmp2->count(),tmp2->mutable_gpu_data(),Layer<float>::patch_mask[1]->mutable_gpu_data());

  delete tmp1; delete tmp2;


}

static void dt_pooling_parameter(MEX_ARGS) {
   mxCHECK( mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
         "Usage: caffe_('blob_set_data', hBlob, new_data)");
 
  const  mxArray* mx_mat1=prhs[1];
  const float* height = reinterpret_cast<const float*>(mxGetData(mx_mat1)); 
  const  mxArray* mx_mat2=prhs[2];
  const float* width = reinterpret_cast<const float*>(mxGetData(mx_mat2));    
  Layer<float>::dt_height.resize(1);
  Layer<float>::dt_height[0].reset(new Blob<float>());
  Layer<float>::dt_height[0]->Reshape(1,1,1,1);
  Layer<float>::dt_width.resize(1); 
  Layer<float>::dt_width[0].reset(new Blob<float>());
  Layer<float>::dt_width[0]->Reshape(1,1,1,1);
  float* dt_height_cpu=Layer<float>::dt_height[0]->mutable_cpu_data();
  float* dt_width_cpu=Layer<float>::dt_width[0]->mutable_cpu_data();
  dt_height_cpu[0]=height[0]; dt_width_cpu[0]=width[0];

//  Layer<float>::seventhlayer_tmp.resize(1);
//  Layer<float>::seventhlayer_tmp[0].reset(new Blob<float>());
//  Layer<float>::seventhlayer_tmp[0]->Reshape(1,1,11*11,dt_height_cpu[0]*dt_width_cpu[0]);  

Blob<float>* template_x=new Blob<float>();
template_x->Reshape(1,1,21,21);
Blob<float>* template_x1=new Blob<float>();
template_x1->Reshape(1,1,21,21);
Blob<float>* template_y=new Blob<float>();
template_y->Reshape(1,1,21,21);
Blob<float>* template_y1=new Blob<float>();
template_y1->Reshape(1,1,21,21);
mx_mat_to_blob(prhs[3],template_x, DATA);
mx_mat_to_blob(prhs[4],template_x1, DATA);
mx_mat_to_blob(prhs[5],template_y, DATA);
mx_mat_to_blob(prhs[6],template_y1, DATA);

Layer<float>::mask_out.resize(1);
    Layer<float>::mask_out[0].reset(new Blob<float>());
    Layer<float>::mask_out[0]->Reshape(1,9,61,61);


Layer<float>::seventhlayer_tmp.resize(1);
Layer<float>::seventhlayer_tmp[0].reset(new Blob<float>());
Layer<float>::seventhlayer_tmp[0]->Reshape(1,1,21*21,dt_height_cpu[0]*dt_width_cpu[0]);

Layer<float>::seventhlayer_tmp1.resize(1);
Layer<float>::seventhlayer_tmp1[0].reset(new Blob<float>());
Layer<float>::seventhlayer_tmp1[0]->Reshape(1,1,4,dt_height_cpu[0]*dt_width_cpu[0]); //按列存储x x^2 y y^2

Layer<float>::seventhlayer_template_x.resize(1);
Layer<float>::seventhlayer_template_x[0].reset(new Blob<float>());
Layer<float>::seventhlayer_template_x[0]->Reshape(1,1,21,21);
Layer<float>::seventhlayer_template_x1.resize(1);
Layer<float>::seventhlayer_template_x1[0].reset(new Blob<float>());
Layer<float>::seventhlayer_template_x1[0]->Reshape(1,1,21,21);
Layer<float>::seventhlayer_template_y.resize(1);
Layer<float>::seventhlayer_template_y[0].reset(new Blob<float>());
Layer<float>::seventhlayer_template_y[0]->Reshape(1,1,21,21);
Layer<float>::seventhlayer_template_y1.resize(1);
Layer<float>::seventhlayer_template_y1[0].reset(new Blob<float>());
Layer<float>::seventhlayer_template_y1[0]->Reshape(1,1,21,21);

Layer<float>::seventhlayer_col_buff.resize(1);
Layer<float>::seventhlayer_col_buff[0].reset(new Blob<float>());
Layer<float>::seventhlayer_col_buff[0]->Reshape(1,1,21*21,dt_height_cpu[0]*dt_width_cpu[0]);


caffe_copy(template_x->count(),template_x->mutable_gpu_data(),Layer<float>::seventhlayer_template_x[0]->mutable_gpu_data());
caffe_copy(template_x1->count(),template_x1->mutable_gpu_data(),Layer<float>::seventhlayer_template_x1[0]->mutable_gpu_data());
caffe_copy(template_y->count(),template_y->mutable_gpu_data(),Layer<float>::seventhlayer_template_y[0]->mutable_gpu_data());
 caffe_copy(template_y1->count(),template_y1->mutable_gpu_data(),Layer<float>::seventhlayer_template_y1[0]->mutable_gpu_data());
delete template_x; delete template_x1; delete template_y; delete template_y1;



}

static void get_tmp(MEX_ARGS) {
    
Blob<float>* tmp=new Blob<float>();
tmp->Reshape(Layer<float>::seventhlayer_tmp1[0]->num(),Layer<float>::seventhlayer_tmp1[0]->channels(),Layer<float>::seventhlayer_tmp1[0]->height(),Layer<float>::seventhlayer_tmp1[0]->width());
Blob<float>* mask_out=new Blob<float>();
mask_out->Reshape(Layer<float>::mask_out[0]->num(),Layer<float>::mask_out[0]->channels(),Layer<float>::mask_out[0]->height(),Layer<float>::mask_out[0]->width());
caffe_copy(tmp->count(),Layer<float>::seventhlayer_tmp1[0]->mutable_gpu_data(),tmp->mutable_gpu_data());
caffe_copy(mask_out->count(),Layer<float>::mask_out[0]->mutable_gpu_data(),mask_out->mutable_gpu_data());
plhs[0] = blob_to_mx_mat(tmp, DATA);
plhs[1] = blob_to_mx_mat(mask_out, DATA);
}


// 我的一些定义
static void im2col(MEX_ARGS) {
    const int* dims = static_cast<const int*>(mxGetDimensions(prhs[1])); 
    Blob<float> im_data(1, 1, dims[1], dims[0]);
    LOG(INFO) << 1 << " " << 1 << " " << dims[1] << " " << dims[0];

    mx_mat_to_blob(prhs[1], &im_data, DATA);
    
    int kernel_h = static_cast<int>(mxGetScalar(prhs[2]));
    int kernel_w = static_cast<int>(mxGetScalar(prhs[3]));
    int stride_h = static_cast<int>(mxGetScalar(prhs[4]));
    int stride_w = static_cast<int>(mxGetScalar(prhs[5]));
    int pad_h = static_cast<int>(mxGetScalar(prhs[6]));
    int pad_w = static_cast<int>(mxGetScalar(prhs[7]));
    int col_dim=kernel_h*kernel_w;
    int out_h = (dims[1] - kernel_h+2*pad_h)/stride_h +1;
    int out_w = (dims[0] - kernel_w+2*pad_w)/stride_w +1;
    
    Blob<float> col_data(1,1,col_dim,out_h*out_w);
    for (int n = 0; n < im_data.num(); n++) {
        switch (Caffe::mode()) {
            case Caffe::CPU: {
            const float* in_cpu = im_data.cpu_data() + im_data.offset(n);
            float* out_cpu = col_data.mutable_cpu_data() + col_data.offset(n);
            im2col_cpu(in_cpu, 1, dims[1], dims[0], kernel_h, kernel_w, pad_h, pad_w, 
                       stride_h, stride_w, out_cpu, 1, 1);
            }
            break;
            case Caffe::GPU: {
            const float* in_gpu = im_data.gpu_data() + im_data.offset(n);
            float* out_gpu = col_data.mutable_gpu_data() + col_data.offset(n);
            im2col_gpu(in_gpu, 1, dims[1], dims[0], kernel_h, kernel_w, pad_h, pad_w, 
                       stride_h, stride_w, out_gpu,1, 1);
            }
            break;
            default:
            mxERROR("Unknown Caffe mode");
        }
    }
    plhs[0] = blob_to_mx_mat(&col_data, DATA);
}






// Usage: caffe_('blob_get_diff', hBlob)
static void blob_get_diff(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('blob_get_diff', hBlob)");
  Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
  plhs[0] = blob_to_mx_mat(blob, DIFF);
}

// Usage: caffe_('blob_set_diff', hBlob, new_diff)
static void blob_set_diff(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsSingle(prhs[1]),
      "Usage: caffe_('blob_set_diff', hBlob, new_diff)");
  Blob<float>* blob = handle_to_ptr<Blob<float> >(prhs[0]);
  mx_mat_to_blob(prhs[1], blob, DIFF);
}

// Usage: caffe_('set_mode_cpu')
static void set_mode_cpu(MEX_ARGS) {
  mxCHECK(nrhs == 0, "Usage: caffe_('set_mode_cpu')");
  Caffe::set_mode(Caffe::CPU);
}

// Usage: caffe_('set_mode_gpu')
static void set_mode_gpu(MEX_ARGS) {
  mxCHECK(nrhs == 0, "Usage: caffe_('set_mode_gpu')");
  Caffe::set_mode(Caffe::GPU);
}

// Usage: caffe_('set_device', device_id)
static void set_device(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsDouble(prhs[0]),
      "Usage: caffe_('set_device', device_id)");
  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id);
}

// Usage: caffe_('get_init_key')
static void get_init_key(MEX_ARGS) {
  mxCHECK(nrhs == 0, "Usage: caffe_('get_init_key')");
  plhs[0] = mxCreateDoubleScalar(init_key);
}

// Usage: caffe_('reset')
static void reset(MEX_ARGS) {
  mxCHECK(nrhs == 0, "Usage: caffe_('reset')");
  // Clear solvers and stand-alone nets
  mexPrintf("Cleared %d solvers and %d stand-alone nets\n",
      solvers_.size(), nets_.size());
  solvers_.clear();
  nets_.clear();
  // Generate new init_key, so that handles created before becomes invalid
  init_key = static_cast<double>(caffe_rng_rand());
}

// Usage: caffe_('reset_by_index', index); index must be an descending order
static void delete_solver(MEX_ARGS) {
  mxCHECK(nrhs == 1, "Usage: caffe_('delete_solver', index)");
  int num_solver = mxGetNumberOfElements(prhs[0]);
  double* index = mxGetPr(prhs[0]);
  for (int i = 0; i < num_solver; i++) {
    LOG(INFO) << "solver size: " << solvers_.size();
    solvers_.erase(solvers_.begin() + static_cast<int>(index[i]) - 1);
  }

  // Clear solvers and stand-alone nets
  mexPrintf("Cleared %d solvers\n", num_solver);
  // Generate new init_key, so that handles created before becomes invalid
  // init_key = static_cast<double>(caffe_rng_rand());
}

// Usage: caffe_('read_mean', mean_proto_file)
static void read_mean(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsChar(prhs[0]),
      "Usage: caffe_('read_mean', mean_proto_file)");
  char* mean_proto_file = mxArrayToString(prhs[0]);
  mxCHECK_FILE_EXIST(mean_proto_file);
  Blob<float> data_mean;
  BlobProto blob_proto;
  bool result = ReadProtoFromBinaryFile(mean_proto_file, &blob_proto);
  mxCHECK(result, "Could not read your mean file");
  data_mean.FromProto(blob_proto);
  plhs[0] = blob_to_mx_mat(&data_mean, DATA);
  mxFree(mean_proto_file);
}

// Usage: caffe_('set_net_phase', hNet, phase_name)
static void set_net_phase(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxIsChar(prhs[1]),
      "Usage: caffe_('set_net_phase', hNet, phase_name)");
  char* phase_name = mxArrayToString(prhs[1]);
  Phase phase;
  if (strcmp(phase_name, "train") == 0) {
    phase = TRAIN;
  } else if (strcmp(phase_name, "test") == 0) {
    phase = TEST;
  } else {
    mxERROR("Unknown phase");
  }
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  net->set_net_phase(phase);
  mxFree(phase_name);
}

// Usage: caffe_('empty_net_param_diff', hNet)
static void empty_net_param_diff(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('empty_net_param_diff', hNet)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  for (int i = 0; i < net->params().size(); ++i) {
    shared_ptr<Blob<float> > blob = net->params()[i];
    switch (Caffe::mode()) {
      case Caffe::CPU:
	caffe_set(blob->count(), static_cast<float>(0),
	    blob->mutable_cpu_diff());
	break;
      case Caffe::GPU:
#ifndef CPU_ONLY
	caffe_gpu_set(blob->count(), static_cast<float>(0),
	    blob->mutable_gpu_diff());
#else
	NO_GPU;
#endif
	break;
    }
  }
}

// Usage: caffe_('apply_update', hSolver)
static void apply_update(MEX_ARGS) {
  mxCHECK(nrhs == 1 && mxIsStruct(prhs[0]),
      "Usage: caffe_('apply_update', hSolver)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  solver->MatCaffeApplyUpdate();
}

// Usage: caffe_('set_input_dim', hNet, dim)
static void set_input_dim(MEX_ARGS) {
  int blob_num = mxGetM(prhs[1]);
  int dim_num = mxGetN(prhs[1]);
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && dim_num == 5,
      "Usage: caffe_('set_input_dim', hNet, dim)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  double* dim = mxGetPr(prhs[1]);
  for (int blob_id = 0; blob_id < blob_num; blob_id++) {
    int id = static_cast<int>(dim[0*blob_num]);
    int n = static_cast<int>(dim[1*blob_num]);
    int c = static_cast<int>(dim[2*blob_num]);
    int h = static_cast<int>(dim[3*blob_num]);
    int w = static_cast<int>(dim[4*blob_num]);
    LOG(INFO) << "Reshape input blob.";
    LOG(INFO) << "Input_id = " << id;
    LOG(INFO) << "num = " << n;
    LOG(INFO) << "channel = " << c;
    LOG(INFO) << "height = " << h;
    LOG(INFO) << "width = " << w;
    net->input_blobs()[id]->Reshape(n, c, h, w);
    dim += 1;
  }
  // Reshape each layer of the network
  net->Reshape();
}

// Usage: caffe_('cnn2fcn', hNet)
static void cnn2fcn(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxGetNumberOfElements(prhs[1]) == 2, "Usage: caffe_('cnn2fcn', hNet, [kstride, pad])");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  double* param = mxGetPr(prhs[1]);
  int kstride = static_cast<int>(param[0]);
  int pad = static_cast<int>(param[1]);

  net->CNN2FCN(kstride, pad);
}

// Usage: caffe_('fcn2cnn', hNet)
static void fcn2cnn(MEX_ARGS) {
  mxCHECK(nrhs == 2 && mxIsStruct(prhs[0]) && mxGetNumberOfElements(prhs[1]) == 1,
      "Usage: caffe_('fcn2cnn', hNet, pad)");
  Net<float>* net = handle_to_ptr<Net<float> >(prhs[0]);
  double* pad_pr = mxGetPr(prhs[1]);
  int pad = static_cast<int>(pad_pr[0]);
  net->FCN2CNN(pad);
}

// Usage: caffe_('snapshot', hSolver, solver_name, model_name)
static void snapshot(MEX_ARGS) {
  mxCHECK(nrhs == 3 && mxIsStruct(prhs[0]),
      "Usage: caffe_('snapshot', hSolver, solver_name, model_name)");
  Solver<float>* solver = handle_to_ptr<Solver<float> >(prhs[0]);
  string solver_name(mxArrayToString(prhs[1]));
  string model_name(mxArrayToString(prhs[2]));
  solver->MatCaffeSnapshot(solver_name, model_name);
}

/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry {
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "get_solver",           get_solver           },
  { "solver_get_attr",      solver_get_attr      },
  { "solver_get_iter",      solver_get_iter      },
  { "solver_restore",       solver_restore       },
  { "solver_solve",         solver_solve         },
  { "solver_step",          solver_step          },
  { "get_net",              get_net              },
  { "net_get_attr",         net_get_attr         },
  { "net_forward",          net_forward          },
  { "net_backward",         net_backward         },
  { "net_copy_from",        net_copy_from        },
  { "net_reshape",          net_reshape          },
  { "net_save",             net_save             },
  { "layer_get_attr",       layer_get_attr       },
  { "layer_get_type",       layer_get_type       },
  { "blob_get_shape",       blob_get_shape       },
  { "set_index",            set_index            },  
  { "blob_reshape",         blob_reshape         },
  { "blob_get_data",        blob_get_data        },
  { "blob_set_data",        blob_set_data        },
  { "my_set_data_a",        my_set_data_a        },
  { "my_set_data_a1",       my_set_data_a1       },
  { "my_set_output_a",      my_set_output_a      },
  { "get_data_from_matlab", get_data_from_matlab }, 
  { "update_samplesf",      update_samplesf      },  
  { "inialize_blobs",       inialize_blobs       },
  { "input_yf",             input_yf             }, 
  { "input_sample_weight",  input_sample_weight  },
  { "set_reg_window",       set_reg_window       },
  { "set_binary_mask_adaptive",       set_binary_mask_adaptive       },
  { "set_hf_4",             set_hf_4             },
  { "set_hf_5",             set_hf_5             },
  { "set_xf_yf4",           set_xf_yf4           },
  { "set_xf_yf5",           set_xf_yf5           },  
  {"get_samplesf",          get_samplesf         },  
  { "set_L_index",          set_L_index          },
  { "clear_memory_function", clear_memory_function},
  { "set_H_4",                set_H_4            },  
  { "set_H_5",                set_H_5            },
  { "set_patch_mask",       set_patch_mask       },
  { "set_frame_id",         set_frame_id         },
  { "dt_pooling_parameter", dt_pooling_parameter },   
  { "get_tmp",              get_tmp              },
  { "im2col",               im2col               },
  { "blob_get_diff",        blob_get_diff        },
  { "blob_set_diff",        blob_set_diff        },
  { "set_mode_cpu",         set_mode_cpu         },
  { "set_mode_gpu",         set_mode_gpu         },
  { "set_device",           set_device           },
  { "get_init_key",         get_init_key         },
  { "reset",                reset                },
  { "delete_solver",        delete_solver        },
  { "read_mean",            read_mean            },
  { "set_net_phase",        set_net_phase        },    
  { "empty_net_param_diff", empty_net_param_diff },
  { "apply_update",         apply_update         },
  { "set_input_dim",        set_input_dim        },
  { "cnn2fcn",              cnn2fcn              },
  { "fcn2cnn",              fcn2cnn              },
  { "snapshot",             snapshot             },
  // The end.
  { "END",                  NULL                 },
};

/** -----------------------------------------------------------------
 ** matlab entry point.
 **/
// Usage: caffe_(api_command, arg1, arg2, ...)
void mexFunction(MEX_ARGS) {
  mexLock();  // Avoid clearing the mex file.
  mxCHECK(nrhs > 0, "Usage: caffe_(api_command, arg1, arg2, ...)");
  // Handle input command
  char* cmd = mxArrayToString(prhs[0]);
  bool dispatched = false;
  // Dispatch to cmd handler
  for (int i = 0; handlers[i].func != NULL; i++) {
    if (handlers[i].cmd.compare(cmd) == 0) {
      handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
      dispatched = true;
      break;
    }
  }
  if (!dispatched) {
    ostringstream error_msg;
    error_msg << "Unknown command '" << cmd << "'";
    mxERROR(error_msg.str().c_str());
  }
  mxFree(cmd);
}
