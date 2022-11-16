#include "iostream"

#include <cublasLt.h>
#include <cublas_v2.h>


// Usage: nvcc -O3 cublaslt_test.cu -lcublasLt -o cublaslt_test

void SetCublasMatrixLayout(cublasLtMatrixLayout_t layout_desc,
                           cublasOperation_t cublas_trans,
                           const size_t cublas_m,
                           const size_t cublas_n) {
    cublasLtMatrixLayoutSetAttribute(
            layout_desc,
            CUBLASLT_MATRIX_LAYOUT_ROWS,
            cublas_trans == CUBLAS_OP_N ? &cublas_m : &cublas_n,
            sizeof(cublas_m));
    cublasLtMatrixLayoutSetAttribute(
            layout_desc,
            CUBLASLT_MATRIX_LAYOUT_COLS,
            cublas_trans == CUBLAS_OP_N ? &cublas_n : &cublas_m,
            sizeof(cublas_m));
    const size_t cublas_ld = cublas_trans == CUBLAS_OP_N ? cublas_m : cublas_n;
    cublasLtMatrixLayoutSetAttribute(
            layout_desc,
            CUBLASLT_MATRIX_LAYOUT_LD,
            &cublas_ld,
            sizeof(cublas_ld));
  }

struct Result {

  double runtime_ms;
  cudaError_t error;
  //
  // Methods
  //
  Result(
    double runtime_ms = 0,
    cudaError_t error = cudaSuccess
  ): runtime_ms(runtime_ms), error(error){ }
};

int main(){
    cudaDataType_t mat_type = CUDA_R_16F;
    cudaDataType_t scale_type = CUDA_R_32F;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

    cublasLtMatmulDesc_t operation_desc_;
    cublasLtMatrixLayout_t x_desc_;
    cublasLtMatrixLayout_t w_desc_;
    cublasLtMatrixLayout_t out_desc_;

    cublasLtMatmulDescCreate(&operation_desc_, compute_type, scale_type);
    cublasLtMatrixLayoutCreate(&x_desc_, mat_type, 1, 1, 1);
    cublasLtMatrixLayoutCreate(&w_desc_, mat_type, 1, 1, 1);
    cublasLtMatrixLayoutCreate(&out_desc_, mat_type, 1, 1, 1);

    // int64_t bsz_seq = 2048; 
    // int64_t in_feature = 16384; 
    // int64_t hidden_feature = 4096; 

    int64_t bsz_seq = 5120; 
    int64_t in_feature = 4096; 
    int64_t hidden_feature = 4096; 

    // int64_t bsz_seq = 4096; 
    // int64_t in_feature = 4096; 
    // int64_t hidden_feature = 5120; 

    int64_t M = bsz_seq;
    int64_t K = in_feature;
    int64_t N = hidden_feature;

    half* w_data; 
    half* x_data; 
    half* out_data; 
    half* bias_data; 
    uint8_t* workspace; 
    size_t workspace_size = 16 * 1024 * 1024; 
    cudaMalloc(&x_data, sizeof(half)* M * K); 
    cudaMalloc(&w_data, sizeof(half)* K * N); 
    cudaMalloc(&out_data, sizeof(half)* M * N); 
    cudaMalloc(&bias_data, sizeof(half)* N); 
    cudaMalloc(&workspace, workspace_size); 

    cublasOperation_t cublas_transA = CUBLAS_OP_N;
    cublasOperation_t cublas_transB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(
            operation_desc_,
            CUBLASLT_MATMUL_DESC_TRANSB,
            &cublas_transA,
            sizeof(cublas_transA));
    cublasLtMatmulDescSetAttribute(
            operation_desc_,
            CUBLASLT_MATMUL_DESC_TRANSA,
            &cublas_transB,
            sizeof(cublas_transB));

    cublasLtEpilogue_t epiloque_func = CUBLASLT_EPILOGUE_GELU_BIAS;
    // cublasLtEpilogue_t epiloque_func = CUBLASLT_EPILOGUE_RELU_BIAS;

    cublasLtMatmulDescSetAttribute(
            operation_desc_,
            CUBLASLT_MATMUL_DESC_EPILOGUE,
            &epiloque_func,
            sizeof(epiloque_func));

    cublasLtMatmulDescSetAttribute(
        operation_desc_,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER,
        &bias_data,
        sizeof(bias_data));

    /*
    cublas use col major: x(M, K) matmul w(K, N) = out(M, N) equals to w_t(N, K)
    * x_t(K, M) = out(N, M)
    */
    SetCublasMatrixLayout(x_desc_, cublas_transA, K, M);
    SetCublasMatrixLayout(w_desc_, cublas_transB, N, K);
    SetCublasMatrixLayout(out_desc_, CUBLAS_OP_N, N, M);

    cudaStream_t stream; 
    cudaStreamCreate(&stream); 

    cublasLtHandle_t handle; 
    cublasLtCreate(&handle); 

    float alpha32 = 1.0f, beta32 = 0.0f;
    void *alpha = nullptr, *beta = nullptr;
    alpha = &alpha32;
    beta = &beta32;

    cublasStatus_t cublas_status; 
    
    cublas_status = cublasLtMatmul(handle,
                                operation_desc_,
                                alpha,
                                w_data,
                                w_desc_,
                                x_data,
                                x_desc_,
                                beta,
                                out_data,
                                out_desc_,
                                out_data,
                                out_desc_,
                                nullptr /*algo*/,
                                workspace /*workspace*/,
                                workspace_size,
                                stream);
    if(cublas_status != CUBLAS_STATUS_SUCCESS){
        printf("Error. \n"); 
    }
    Result profile_result;

    cudaEvent_t events[2];

    for (auto & event : events) {
        profile_result.error = cudaEventCreate(&event);
        if (profile_result.error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(profile_result.error) << std::endl;
        return -1;
        }
    }

    // Record an event at the start of a series of GEMM operations
    profile_result.error = cudaEventRecord(events[0]);
    if (profile_result.error != cudaSuccess) {
        std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(profile_result.error) << std::endl;
        return -1; 
    }

    //
    // Run profiling loop
    //
    const int32_t iter_num = 1; 

    for (int iter = 0; iter < iter_num; ++iter) {
        cublasLtMatmul(handle,
                        operation_desc_,
                        alpha,
                        w_data,
                        w_desc_,
                        x_data,
                        x_desc_,
                        beta,
                        out_data,
                        out_desc_,
                        out_data,
                        out_desc_,
                        nullptr /*algo*/,
                        workspace /*workspace*/,
                        workspace_size,
                        stream);
        if(cublas_status != CUBLAS_STATUS_SUCCESS){
            printf("Error. \n"); 
        }
    }

    profile_result.error = cudaEventRecord(events[1]);
    if (profile_result.error != cudaSuccess) {
        std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(profile_result.error) << std::endl;
        return -1; 
    }

    // Wait for work on the device to complete.
    profile_result.error = cudaEventSynchronize(events[1]);
    if (profile_result.error != cudaSuccess) {
        std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(profile_result.error) << std::endl;
        return -1;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    profile_result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (profile_result.error != cudaSuccess) {
        std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(profile_result.error) << std::endl;
        return -1;
    }

    // Compute average runtime and GFLOPs.
    profile_result.runtime_ms = double(runtime_ms) / double(iter_num);
    
    //
    // Cleanup
    //

    for (auto event : events) {
        (void)cudaEventDestroy(event);
    }

    cudaDeviceSynchronize();

    std::cout << std::endl;
    std::cout << "cublasLt Gemm+Bias+GELU:\n"
        << "====================================================" << std::endl;
    std::cout << "    " << " {M, K, N} = {" << M \
        << ", " << K << ", " << N <<"}." << std::endl;
    std::cout << std::endl;
    std::cout << "    " << "Runtime: " << profile_result.runtime_ms << " ms" << std::endl;

    cudaFree(x_data); 
    cudaFree(w_data); 
    cudaFree(out_data); 
    cudaFree(bias_data); 
    cudaFree(workspace); 
    cudaStreamDestroy(stream); 
    cublasLtDestroy(handle); 
    return 0;
}   