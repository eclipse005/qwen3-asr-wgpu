// Dump every intermediate of CUDA's expf sequence (manually expanded, exactly
// mirroring the sm_61 PTX emitted for expf by CUDA 12.8) to bisect the WGSL port.
extern "C" __global__ void probe2(
    const float* __restrict__ in_x,
    float* __restrict__ out,   // 8 intermediates + final per input
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in_x[i];

    float f4 = fmaf(x, __uint_as_float(0x3BBB989Du), __uint_as_float(0x3F000000u));
    float f5 = fmin(fmax(f4, 0.0f), 1.0f);   // cvt.sat.f32.f32
    float f8 = __fmaf_rd(f5, __uint_as_float(0x437C0000u), __uint_as_float(0x4B400001u));
    float f9 = f8 + __uint_as_float(0xCB40007Fu);
    float f10 = -f9;
    float f12 = fmaf(x, __uint_as_float(0x3FB8AA3Bu), f10);
    float f14 = fmaf(x, __uint_as_float(0x32A57060u), f12);
    unsigned int r6 = __float_as_uint(f8) << 23;
    float f15 = __uint_as_float(r6);
    float f16;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(f16) : "f"(f14));
    float fin = f16 * f15;

    out[i*12+0] = f5;
    out[i*12+1] = f8;
    out[i*12+2] = f12;
    out[i*12+3] = f14;
    out[i*12+4] = f15;
    out[i*12+5] = f16;
    out[i*12+6] = fin;
    out[i*12+7] = expf(x);
    out[i*12+8] = fmaf(f5, 251.0f, 12582913.0f);              // round-to-nearest fused
    out[i*12+9] = f5 * 251.0f + 12582913.0f;                  // source mul+add (nvcc contracts)
    out[i*12+10] = fmaf(f5, 251.0f, 12582913.0f) - fmaf(f5, 251.0f, 12582913.0f); // sanity 0
}
