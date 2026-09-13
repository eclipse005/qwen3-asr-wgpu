// PTX probe: what does CUDA generate for the transcendentals the decode path uses?
// Three独立输入数组, outputs f32-as-u32 bits for exact comparison with the WGSL side.
extern "C" __global__ void probe(
    const float* __restrict__ in_x,
    const float* __restrict__ in_y,
    const float* __restrict__ in_z,
    unsigned int* __restrict__ out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in_x[i];
    float y = in_y[i];
    float z = in_z[i];
    out[i*4+0] = __float_as_uint(expf(x));       // exact exp (attention softmax)
    out[i*4+1] = __float_as_uint(fmaf(x, y, z)); // fused fma
    out[i*4+2] = __float_as_uint(__expf(x));     // fast exp (silu path)
    out[i*4+3] = __float_as_uint(x * y + z);     // separate mul+add (contract check baseline)
}
