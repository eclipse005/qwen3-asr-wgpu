//! cuBLAS prefill-GEMM baseline, for direct comparison against the WGSL GEMM in the
//! sibling wgpu spike.  Same shapes, same f16 weights, same f32 accumulate.
//!
//! C[m,n] = A[m,k] * W[n,k]^T  (row-major throughout; cuBLAS is called column-major
//! with the transa=T / transb=N trick, exactly like examples/cublas_gemv_bench.rs).
//!
//! Usage: cargo run --release

use std::time::Instant;

use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublas::sys;
use cudarc::driver::CudaContext;
use half::f16;

const HS: usize = 1024;
const Q_DIM: usize = 16 * 128;                    // 2048
const KV_DIM: usize = 8 * 128;                    // 1024
const FUSED_QKV_COLS: usize = Q_DIM + 2 * KV_DIM; // 4096
const INTER: usize = 3072;
const FUSED_GU_COLS: usize = 2 * INTER;           // 6144

/// (label, m, n, k) — n is the weight's output dim, k its input dim.
const SHAPES: &[(&str, usize, usize, usize)] = &[
    ("qkv",     384, FUSED_QKV_COLS, HS),
    ("o_proj",  384, HS,             Q_DIM),
    ("gate_up", 384, FUSED_GU_COLS,  HS),
    ("down",    384, HS,             INTER),
    ("qkv",     1536, FUSED_QKV_COLS, HS),
    ("gate_up", 1536, FUSED_GU_COLS,  HS),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone())?;
    unsafe {
        sys::cublasSetMathMode(*blas.handle(), sys::cublasMath_t::CUBLAS_TENSOR_OP_MATH);
    }

    println!("=== cuBLAS prefill GEMM baseline  C[m,n] = A[m,k] * W[n,k]^T ===");
    println!("{:<10} {:>6} {:>7} {:>6} {:>10} {:>11} {:>9}",
        "shape", "m", "n", "k", "ms", "GFLOP/s", "GB/s(W)");
    println!("{}", "-".repeat(63));

    for &(label, m, n, k) in SHAPES {
        let a_host = vec![f16::from_f32(0.75); m * k];
        let w_host = vec![f16::from_f32(0.75); n * k];
        let a = stream.clone_htod(&a_host)?;
        let w = stream.clone_htod(&w_host)?;
        let mut c = stream.alloc_zeros::<f16>(m * n)?;

        let cfg = GemmConfig {
            transa: sys::cublasOperation_t::CUBLAS_OP_T,
            transb: sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,          // n_out
            n: m as i32,          // m_in
            k: k as i32,
            alpha: f16::from_f32(1.0),
            lda: k as i32,
            ldb: k as i32,
            ldc: n as i32,
            beta: f16::from_f32(0.0),
        };

        for _ in 0..3 {
            unsafe { blas.gemm(cfg, &w, &a, &mut c)?; }
        }
        stream.synchronize()?;

        let iters = 20;
        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe { blas.gemm(cfg, &w, &a, &mut c)?; }
        }
        stream.synchronize()?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let wbytes = (n * k * 2) as f64;
        println!("{:<10} {:>6} {:>7} {:>6} {:>10.3} {:>11.1} {:>9.1}",
            label, m, n, k, ms,
            flops / (ms / 1000.0) / 1e9,
            wbytes / 1e9 / (ms / 1000.0));
    }

    Ok(())
}
