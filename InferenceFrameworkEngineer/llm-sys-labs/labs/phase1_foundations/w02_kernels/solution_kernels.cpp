// Week 2 (CPU path): the same three algorithms written in the CUDA
// execution model's shape, so the exercise transfers 1:1 to the .cu file.
//
//   CUDA concept        CPU twin here
//   grid of blocks   -> OpenMP parallel loop over "blocks"
//   threads in block -> inner loop over "threads" (vectorised by the compiler)
//   __shared__ tile  -> small stack array that fits in L1
//
// Build is automatic from solution.py (g++ -O3 -fopenmp -march=native).
#include <torch/extension.h>
#include <omp.h>
#include <vector>

#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda() && x.is_contiguous() && x.dtype() == torch::kFloat32, #x " must be contiguous CPU float32")

constexpr int TILE = 64;  // 3 tiles x 64x64 x 4 B = 48 KB: roughly an L1/L2 working set

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    CHECK_CPU(a); CHECK_CPU(b);
    auto c = torch::empty_like(a);
    const float* pa = a.data_ptr<float>();
    const float* pb = b.data_ptr<float>();
    float* pc = c.data_ptr<float>();
    int64_t n = a.numel();
    // >>> SOLUTION: parallel-for over elements (each iteration = one CUDA thread)
#pragma omp parallel for simd
    for (int64_t i = 0; i < n; ++i) pc[i] = pa[i] + pb[i];
    // <<< SOLUTION
    return c;
}

torch::Tensor reduce_sum(torch::Tensor x) {
    CHECK_CPU(x);
    const float* p = x.data_ptr<float>();
    int64_t n = x.numel();
    const int64_t block = 4096;  // elements per "block"
    int64_t nblocks = (n + block - 1) / block;
    std::vector<double> partial(nblocks, 0.0);
    double total = 0.0;
    // >>> SOLUTION: stage 1 - each block sums its slice into partial[b]; stage 2 - sum the partials into total
#pragma omp parallel for
    for (int64_t b = 0; b < nblocks; ++b) {
        double s = 0.0;
        int64_t hi = std::min(n, (b + 1) * block);
        for (int64_t i = b * block; i < hi; ++i) s += p[i];
        partial[b] = s;
    }
    for (double s : partial) total += s;
    // <<< SOLUTION
    return torch::tensor((float)total);
}

torch::Tensor sgemm(torch::Tensor A, torch::Tensor B) {
    CHECK_CPU(A); CHECK_CPU(B);
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "shape mismatch");
    auto C = torch::zeros({M, N}, A.options());
    const float* pa = A.data_ptr<float>();
    const float* pb = B.data_ptr<float>();
    float* pc = C.data_ptr<float>();
    const int mt = (M + TILE - 1) / TILE, nt = (N + TILE - 1) / TILE;
#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < mt; ++bi) {
        for (int bj = 0; bj < nt; ++bj) {
            alignas(64) float As[TILE][TILE];
            alignas(64) float Bs[TILE][TILE];
            alignas(64) float Cs[TILE][TILE] = {};
            for (int k0 = 0; k0 < K; k0 += TILE) {
                // >>> SOLUTION: copy the A and B tiles (zero-pad edges), then accumulate Cs += As @ Bs with j innermost
                for (int i = 0; i < TILE; ++i)
                    for (int k = 0; k < TILE; ++k) {
                        int r = bi * TILE + i, kk = k0 + k;
                        As[i][k] = (r < M && kk < K) ? pa[(int64_t)r * K + kk] : 0.f;
                    }
                for (int k = 0; k < TILE; ++k)
                    for (int j = 0; j < TILE; ++j) {
                        int kk = k0 + k, c = bj * TILE + j;
                        Bs[k][j] = (kk < K && c < N) ? pb[(int64_t)kk * N + c] : 0.f;
                    }
                for (int i = 0; i < TILE; ++i)
                    for (int k = 0; k < TILE; ++k) {
                        const float a = As[i][k];
#pragma omp simd
                        for (int j = 0; j < TILE; ++j) Cs[i][j] += a * Bs[k][j];
                    }
                // <<< SOLUTION
            }
            for (int i = 0; i < TILE; ++i) {
                int r = bi * TILE + i;
                if (r >= M) break;
                for (int j = 0; j < TILE; ++j) {
                    int c = bj * TILE + j;
                    if (c < N) pc[(int64_t)r * N + c] = Cs[i][j];
                }
            }
        }
    }
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add);
    m.def("reduce_sum", &reduce_sum);
    m.def("sgemm", &sgemm);
}
