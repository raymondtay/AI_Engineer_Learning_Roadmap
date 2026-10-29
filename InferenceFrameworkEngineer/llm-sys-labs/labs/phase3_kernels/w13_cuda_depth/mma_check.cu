// Week 13 (GPU only, sm_80+): one warp runs a real mma.sync.m16n8k16 and
// writes D back using the fragment layouts from the PTX ISA tables - the same
// formulas as a_frag / b_frag / c_frag in solution.py. If your Python layout
// disagrees with this file, one of them is wrong; the hardware decides which.
//
// STATUS: not executed by the author (no GPU in the build environment).
#include <torch/extension.h>
#include <cuda_fp16.h>

__device__ __forceinline__ void a_rc(int lane, int i, int& r, int& c) {
    int g = lane >> 2, t = lane & 3;
    r = g + 8 * ((i >> 1) & 1);
    c = t * 2 + (i & 1) + 8 * (i >> 2);
}
__device__ __forceinline__ void b_rc(int lane, int i, int& k, int& n) {
    int g = lane >> 2, t = lane & 3;
    k = t * 2 + (i & 1) + 8 * (i >> 1);
    n = g;
}
__device__ __forceinline__ void c_rc(int lane, int i, int& r, int& c) {
    int g = lane >> 2, t = lane & 3;
    r = g + 8 * (i >> 1);
    c = t * 2 + (i & 1);
}

__global__ void mma_kernel(const half* A, const half* B, const float* C, float* D) {
    int lane = threadIdx.x;
    half a[8], b[4];
    float c[4], d[4];
    for (int i = 0; i < 8; ++i) { int r, cc; a_rc(lane, i, r, cc); a[i] = A[r * 16 + cc]; }
    for (int i = 0; i < 4; ++i) { int k, n; b_rc(lane, i, k, n); b[i] = B[k * 8 + n]; }
    for (int i = 0; i < 4; ++i) { int r, cc; c_rc(lane, i, r, cc); c[i] = C[r * 8 + cc]; }
    const unsigned* ua = reinterpret_cast<const unsigned*>(a);
    const unsigned* ub = reinterpret_cast<const unsigned*>(b);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(ua[0]), "r"(ua[1]), "r"(ua[2]), "r"(ua[3]), "r"(ub[0]), "r"(ub[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    for (int i = 0; i < 4; ++i) { int r, cc; c_rc(lane, i, r, cc); D[r * 8 + cc] = d[i]; }
}

torch::Tensor mma_16x8x16(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kHalf && A.size(0) == 16 && A.size(1) == 16);
    TORCH_CHECK(B.is_cuda() && B.dtype() == torch::kHalf && B.size(0) == 16 && B.size(1) == 8);
    auto Ac = A.contiguous(), Bc = B.contiguous(), Cc = C.contiguous();
    auto D = torch::empty_like(Cc);
    mma_kernel<<<1, 32>>>(reinterpret_cast<const half*>(Ac.data_ptr<at::Half>()),
                          reinterpret_cast<const half*>(Bc.data_ptr<at::Half>()),
                          Cc.data_ptr<float>(), D.data_ptr<float>());
    return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mma_16x8x16", &mma_16x8x16); }
