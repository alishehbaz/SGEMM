#include<cstdio>
#include<cuda_runtime.h>
#include<string>

#define CEIL_DIV(x,y) ((x+((y)-1))/y)

#define CUDA_CHECK(call){                                                           \
    cudaError_t err = call;                                                         \
    if (err != cudaSuccess){                                                        \
        fprintf(stderr, "CUDA error in %s at line %d: %s \n", __FILE__, __LINE__,   \
        cudaGetErrorString(err));                                                   \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
}


// TODO: write the kernel that is responsible for actual multiplication

int main(int argc, const char *argv[]){

    if (argc!=4){
        fprintf(stderr, "ERROR: missing parameters \n");
        fprintf(stderr, "Usage: %s <M> <N> <K> \n", argv[0]);
        return 1;
    }

    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int K = std::stoi(argv[3]);


    float alpha = 1.0;
    float beta = 1.0;

    float *h_A = new float[M*K];
    float *h_B = new float[N*K];
    float *h_C = new float[M*N];

    for (int i=0; i<M*K; i++)   h_A[i] = static_cast<float>(rand())/RAND_MAX;
    for (int i=0; i<N*K; i++)   h_B[i] = static_cast<float>(rand())/RAND_MAX;
    for (int i=0; i<M*N; i++)   h_C[i] = 0.0f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    dim3 blockDim(32, 32, 1);

    CUDA_CHECK(cudaEventRecord(start));

    // launch the kernel
    // kernell call will go here
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    const float gflops = ((float) M*N * (2*K-1)) / (1e6 * ms);
    printf("SGEMM execution time: %.2f ms, %.1f GFLOPs/s\n", ms, gflops);

    printf("MatMul in CUDA \n");
    return 0;


    // free up the host (cpu) memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    // free up the device (gpu) memory and events
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;

}



// terminology =>
// host: cpu
// device: gpu
// core: A processing unit
// block: collection/group of threads
//     usually 1024 threads are present in a block,
//     you define the dimensions using a variable called blockDim 
//     blockDim(32, 32, 1) => 1024 threads
