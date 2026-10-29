// AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
// fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
// Week 2 (GPU path): CUDA C++ kernels. Built by solution.py via
// torch.utils.cpp_extension.load when a CUDA device is present.
//
// Three kernels, in order of difficulty:
//   1. vector_add      - one thread per element, coalesced access
//   2. reduce_sum      - shared-memory tree within a block + warp shuffles
//   3. sgemm_tiled     - 32x32 shared-memory tiles (the classic first GEMM)
//
// Profile with:  ncu --set full python solution.py --bench
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dtype() == torch::kFloat32, #x " must be contiguous CUDA float32")

constexpr int TILE = 32;

__global__ void vector_add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                  float* __restrict__ c, int64_t n) {
    // TODO(w02_kernels): global index = blockIdx.x * blockDim.x + threadIdx.x; guard against n
}

__inline__ __device__ float warp_sum(float v) {
    // TODO(w02_kernels): butterfly with __shfl_down_sync over offsets 16, 8, 4, 2, 1
}

// Each block reduces blockDim.x * 2 elements into partial[blockIdx.x].
__global__ void reduce_sum_kernel(const float* __restrict__ x, float* __restrict__ partial, int64_t n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int64_t i = (int64_t)blockIdx.x * blockDim.x * 2 + tid;
    float v = 0.f;
    if (i < n) v += x[i];
    if (i + blockDim.x < n) v += x[i + blockDim.x];  // first add during load halves idle threads
    // TODO(w02_kernels): warp-reduce, write one value per warp to smem, then warp 0 reduces those
}

__global__ void sgemm_tiled_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                   float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE + 1];  // +1 pad: avoids bank conflicts on column reads
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        // TODO(w02_kernels): cooperatively load one A tile and one B tile (zero-pad edges), sync, multiply, sync
    }
    if (row < M && col < N) C[(int64_t)row * N + col] = acc;
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    CHECK_CUDA(a); CHECK_CUDA(b);
    auto c = torch::empty_like(a);
    int64_t n = a.numel();
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    vector_add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    return c;
}

torch::Tensor reduce_sum(torch::Tensor x) {
    CHECK_CUDA(x);
    int threads = 256;
    auto cur = x.flatten();
    // Repeat block reductions until one value remains.
    while (cur.numel() > 1) {
        int64_t n = cur.numel();
        int blocks = (int)((n + threads * 2 - 1) / (threads * 2));
        auto partial = torch::empty({blocks}, x.options());
        reduce_sum_kernel<<<blocks, threads, (threads / 32) * sizeof(float)>>>(cur.data_ptr<float>(), partial.data_ptr<float>(), n);
        cur = partial;
    }
    return cur.reshape({});
}

torch::Tensor sgemm(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A); CHECK_CUDA(B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "shape mismatch");
    auto C = torch::empty({M, N}, A.options());
    dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    sgemm_tiled_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add);
    m.def("reduce_sum", &reduce_sum);
    m.def("sgemm", &sgemm);
}
