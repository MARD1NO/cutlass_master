// Modify from example 12 gemm_bias_relu 

// #include <algorithem>

#include <iostream>
#include<math.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"
// #include "cutlass/epilogue/thread/linear_combination_relu.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

using ElementAccumulator = float; 
using ElementComputeEpilogue = ElementAccumulator; 
using ElementInputA = cutlass::half_t; 
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;

using LayoutInputA = cutlass::layout::ColumnMajor; 
using LayoutInputB = cutlass::layout::ColumnMajor; 
using LayoutOutput = cutlass::layout::ColumnMajor; 

using MMAOp = cutlass::arch::OpClassTensorOp; 
using SmArch = cutlass::arch::Sm80;


// using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>; 
// using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>; 
// using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>; 


// TileDescription([128, 128, 32], 5, [2, 2, 1], math_inst, min_cc, max_cc)
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>; 
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>; 
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>; 

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>; 
// using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle; 
// using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<8>; 


using EpilogueOp = cutlass::epilogue::thread::LinearCombinationFastGELU<
    ElementOutput, 
    128 / cutlass::sizeof_bits<ElementOutput>::value, 
    ElementAccumulator, 
    ElementComputeEpilogue, 
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>; 

// using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
//     ElementOutput, 
//     128 / cutlass::sizeof_bits<ElementOutput>::value, 
//     ElementAccumulator, 
//     ElementComputeEpilogue, 
//     cutlass::epilogue::thread::ScaleType::NoBetaScaling>; 

constexpr int NumStages = 5; 

using Gemm = cutlass::gemm::device::Gemm<ElementInputA, 
                                         LayoutInputA, 
                                         ElementInputB, 
                                         LayoutInputB, 
                                         ElementOutput, 
                                         LayoutOutput, 
                                         ElementAccumulator, 
                                         MMAOp, 
                                         SmArch, 
                                         ShapeMMAThreadBlock, 
                                         ShapeMMAWarp, 
                                         ShapeMMAOp, 
                                         EpilogueOp, 
                                         SwizzleThreadBlock, 
                                         NumStages,
                                         8/*kAlignmentA*/, 
                                         8/*kAlignmentB*/, 
                                         false/*bool SplitKSerial = false,*/>; 

ElementAccumulator host_gelu_func(ElementAccumulator val){
  const ElementAccumulator temp = erf(val * static_cast<ElementAccumulator>(M_SQRT1_2));
  const ElementAccumulator out = (val * static_cast<ElementAccumulator>(0.5) * (static_cast<ElementOutput>(1) + temp));
  return out; 
}

struct Result {

  double runtime_ms;
  cutlass::Status status;
  cudaError_t error;
  //
  // Methods
  //
  Result(
    double runtime_ms = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ): runtime_ms(runtime_ms), status(status), error(error){ }
};

int run() {
  // const int length_m = 2048; 
  // const int length_n = 4096; 
  // const int length_k = 16384; 

  // const int length_m = 5120;
  // const int length_n = 4096;
  // const int length_k = 4096;

  const int length_m = 4096;
  const int length_n = 5120;
  const int length_k = 4096;

  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k); 

  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
    problem_size.mk()); 
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
    problem_size.kn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
    problem_size.mn());  
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c_bias(
    {problem_size.m(), 1});  
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
    problem_size.mn());  
  
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
    problem_size.mn());  
  
  cutlass::reference::host::TensorFillRandomUniform(
    tensor_a.host_view(), 
    1, 
    ElementInputA(4), 
    ElementInputA(-4),
    0); 
  cutlass::reference::host::TensorFillRandomUniform(
    tensor_b.host_view(), 
    1, 
    ElementInputB(4), 
    ElementInputB(-4),
    0); 
  cutlass::reference::host::TensorFillRandomUniform(
    tensor_c_bias.host_view(), 
    1, 
    ElementOutput(4), 
    ElementOutput(-4),
    0);

  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

  tensor_a.sync_device(); 
  tensor_b.sync_device(); 
  tensor_c_bias.sync_device(); 
  tensor_d.sync_device(); 
  tensor_ref_d.sync_device();   

  ElementComputeEpilogue alpha = ElementComputeEpilogue(1); 

  int split_k_slices = 1; 

  typename Gemm::Arguments arguments{
    problem_size, 
    tensor_a.device_ref(), 
    tensor_b.device_ref(), 
    {tensor_c_bias.device_data(), 0}, 
    tensor_d.device_ref(), 
    {alpha}, 
    split_k_slices}; 

  size_t workspace_size = Gemm::get_workspace_size(arguments); 

  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size); 

  Gemm gemm_op; 

  cutlass::Status status = gemm_op.can_implement(arguments); 
  CUTLASS_CHECK(status); 

  status = gemm_op.initialize(arguments, workspace.get()); 
  CUTLASS_CHECK(status); 

  status = gemm_op(); 
  CUTLASS_CHECK(status); 

  cutlass::reference::device::Gemm<ElementInputA, 
                                   LayoutInputA, 
                                   ElementInputB, 
                                   LayoutInputB, 
                                   ElementOutput, 
                                   LayoutOutput, 
                                   ElementComputeEpilogue, 
                                   ElementComputeEpilogue>
    gemm_device_reference; 

  gemm_device_reference(
    problem_size, 
    alpha, 
    tensor_a.device_ref(), 
    tensor_b.device_ref(), 
    0, 
    tensor_ref_d.device_ref()); 

  cudaDeviceSynchronize(); 

  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  for(int i = 0; i < problem_size.m(); ++i){
    for(int j = 0; j < problem_size.n(); ++j){
      tensor_ref_d.at({i, j}) = host_gelu_func(ElementOutput(tensor_ref_d.at({i, j}) + tensor_c_bias.at({i, 0}))); 
    }
  }

  std::cout << (cutlass::reference::host::TensorEquals(tensor_d.host_view(),
                                                       tensor_ref_d.host_view())
                    ? "Passed"
                    : "Failed")
            << std::endl;

  CUTLASS_CHECK(status);

  // Profile. 
  //
  // Warm-up run of the grouped GEMM object
  //

  gemm_op(); 
  
  //
  // Construct events
  //

  // Result profile_result;

  // cudaEvent_t events[2];

  // for (auto & event : events) {
  //   profile_result.error = cudaEventCreate(&event);
  //   if (profile_result.error != cudaSuccess) {
  //     std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(profile_result.error) << std::endl;
  //     return -1;
  //   }
  // }

  // // Record an event at the start of a series of GEMM operations
  // profile_result.error = cudaEventRecord(events[0]);
  // if (profile_result.error != cudaSuccess) {
  //   std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(profile_result.error) << std::endl;
  //   return -1; 
  // }

  // //
  // // Run profiling loop
  // //
  // const int32_t iter_num = 1; 
  // for (int iter = 0; iter < iter_num; ++iter) {
  //   gemm_op(); 
  // }

  // //
  // // Stop profiling loop
  // //

  // // Record an event when the GEMM operations have been launched.
  // profile_result.error = cudaEventRecord(events[1]);
  // if (profile_result.error != cudaSuccess) {
  //   std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(profile_result.error) << std::endl;
  //   return -1; 
  // }

  // // Wait for work on the device to complete.
  // profile_result.error = cudaEventSynchronize(events[1]);
  // if (profile_result.error != cudaSuccess) {
  //   std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(profile_result.error) << std::endl;
  //   return -1;
  // }

  // // Measure elapsed runtime
  // float runtime_ms = 0;
  // profile_result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  // if (profile_result.error != cudaSuccess) {
  //   std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(profile_result.error) << std::endl;
  //   return -1;
  // }

  // // Compute average runtime and GFLOPs.
  // profile_result.runtime_ms = double(runtime_ms) / double(iter_num);
  
  // //
  // // Cleanup
  // //

  // for (auto event : events) {
  //   (void)cudaEventDestroy(event);
  // }

  // std::cout << std::endl;
  // std::cout << "CUTLASS Gemm+Bias+GELU:\n"
  //   << "====================================================" << std::endl;
  // std::cout << "    " << " {M, K, N} = {" << length_m \
  //   << ", " << length_k << ", " << length_n <<"}." << std::endl;
  // std::cout << std::endl;
  // std::cout << "    " << "Runtime: " << profile_result.runtime_ms << " ms" << std::endl;
  return 0;

}

int main() {

  bool notSupported = false;

  // Turing Tensor Core operations exposed with mma.sync are first available in CUDA 10.2.
  //
  // CUTLASS must be compiled with CUDA 10.1 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
    std::cerr << "Turing Tensor Core operations must be compiled with CUDA 10.2 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (!(props.major * 10 + props.minor >= 75)) {
    std::cerr << "Turing Tensor Ops must be run on a machine with compute capability at least 75."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  return run();
}
