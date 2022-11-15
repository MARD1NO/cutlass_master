// Modify from example 12 gemm_bias_relu 

// #include <algorithem>

#include <iostream>
#include<math.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"
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
using ElementOutput = float; 

using LayoutInputA = cutlass::layout::ColumnMajor; 
using LayoutInputB = cutlass::layout::ColumnMajor; 
using LayoutOutput = cutlass::layout::ColumnMajor; 

using MMAOp = cutlass::arch::OpClassTensorOp; 
using SmArch = cutlass::arch::Sm80;


using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>; 
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>; 
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>; 

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; 

using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGELU<
    ElementOutput, 
    128 / cutlass::sizeof_bits<ElementOutput>::value, 
    ElementAccumulator, 
    ElementComputeEpilogue, 
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>; 


constexpr int NumStages = 2; 

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
                                         NumStages>; 

ElementOutput host_gelu_func(ElementOutput val){
  const ElementOutput temp = erf(val * static_cast<ElementOutput>(M_SQRT1_2));
  const ElementOutput out = (val * static_cast<ElementOutput>(0.5) * (static_cast<ElementOutput>(1) + temp));
  return out; 
}

int run() {
  const int length_m = 5120; 
  const int length_n = 4096; 
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
