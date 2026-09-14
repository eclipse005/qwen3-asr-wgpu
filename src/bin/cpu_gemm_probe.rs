//! Isolated CPU f32 GEMM probe for the audio-encoder conv shapes.
//!
//! The conv stem dominates CPU encode time; this measures what the `gemm`
//! crate actually achieves on the exact (m, n, k) the stem uses, split by
//! tile size, so "the GEMM is slow" can be separated from "im2col is slow".
//!
//! ```text
//! cargo run --release --bin cpu_gemm_probe
//! ```

use std::time::Instant;

use gemm::{gemm, Parallelism};
use rayon::prelude::*;

fn bench(m: usize, k: usize, n: usize, reps: usize) -> f64 {
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.01 - 0.08).collect();
    let b: Vec<f32> = (0..n * k).map(|i| ((i % 13) as f32) * 0.01 - 0.06).collect();
    let mut c = vec![0.0f32; m * n];
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        unsafe {
            gemm(
                m, n, k,
                c.as_mut_ptr(), 1, n as isize,
                false,
                a.as_ptr(), 1, k as isize,
                b.as_ptr(), k as isize, 1,
                0.0, 1.0, false, false, false,
                Parallelism::Rayon(0),
            );
        }
        let e = t.elapsed().as_secs_f64();
        if e < best { best = e; }
    }
    let flop = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = flop / best / 1e12;
    println!(
        "  m={m:<6} n={n:<6} k={k:<6}  {:>8.2} ms  {:>6.2} TFLOP/s  ({:.1} GFLOP/s)",
        best * 1e3,
        tflops,
        tflops * 1000.0
    );
    tflops
}

fn bench_im2col(b: usize, c_in: usize, h: usize, w: usize) -> f64 {
    let x: Vec<f32> = (0..b * c_in * h * w).map(|i| (i % 23) as f32 * 0.01).collect();
    let h_out = (h + 2 - 3) / 2 + 1;
    let w_out = (w + 2 - 3) / 2 + 1;
    let k = c_in * 9;
    let plane = h_out * w_out;
    let mut cols = vec![0.0f32; b * plane * k];
    let t = Instant::now();
    cols.par_chunks_mut(k).enumerate().for_each(|(col_idx, col)| {
        let ib = col_idx / plane;
        let rem = col_idx % plane;
        let ho = rem / w_out;
        let wo = rem % w_out;
        for ic in 0..c_in {
            for kh in 0..3 {
                for kw in 0..3 {
                    let ih = (ho * 2 + kh) as isize - 1;
                    let iw = (wo * 2 + kw) as isize - 1;
                    let v = if ih < 0 || ih >= h as isize || iw < 0 || iw >= w as isize {
                        0.0
                    } else {
                        x[((ib * c_in + ic) * h + ih as usize) * w + iw as usize]
                    };
                    col[ic * 9 + kh * 3 + kw] = v;
                }
            }
        }
    });
    let e = t.elapsed().as_secs_f64();
    let gb = (cols.len() * 4) as f64 / 1e9;
    println!("  im2col b={b} c_in={c_in} {h}x{w} -> {:>8.2} ms  {:.1} GB/s written", e * 1e3, gb / e);
    e
}

/// Reproduce the real `CpuConvStem::conv_block` loop exactly (same Vec sizes,
/// same zero-init, same epilogue) so the difference from `bench()` is visible.
fn loop_probe() {
    for tile in [8usize, 16, 32, 72] {
        let b = 177usize;
        let c_in = 480usize;
        let (h, w) = (32usize, 50usize);
        let c_out = 480usize;
        let (h_out, w_out) = (16usize, 25usize);
        let plane = h_out * w_out;
        let k = c_in * 9;
        let x: Vec<f32> = (0..b * c_in * h * w).map(|i| (i % 23) as f32 * 0.01).collect();
        let w_w: Vec<f32> = (0..c_out * k).map(|i| (i % 11) as f32 * 0.01).collect();
        let bias = vec![0.01f32; c_out];
        let mut out4d = vec![0.0f32; b * c_out * plane];

        let t = Instant::now();
        for b0 in (0..b).step_by(tile) {
            let nb = tile.min(b - b0);
            let (cols, _ho, _wo) = im2col(&x[b0 * c_in * h * w..(b0 + nb) * c_in * h * w], nb, c_in, h, w);
            let col_count = nb * plane;
            let mut gemm_out = vec![0.0f32; c_out * col_count];
            unsafe {
                gemm(
                    c_out, col_count, k,
                    gemm_out.as_mut_ptr(), 1, col_count as isize,
                    false, w_w.as_ptr(), 1, k as isize, cols.as_ptr(), k as isize, 1,
                    0.0, 1.0, false, false, false, Parallelism::Rayon(0),
                );
            }
            let dst = &mut out4d[b0 * c_out * plane..(b0 + nb) * c_out * plane];
            dst.par_chunks_mut(plane).enumerate().for_each(|(pi, pd)| {
                let ib = pi / c_out;
                let oc = pi % c_out;
                let src_row = oc * col_count + ib * plane;
                for i in 0..plane { pd[i] = gemm_out[src_row + i] + bias[oc]; }
            });
        }
        let e = t.elapsed().as_secs_f64();
        let flop = 2.0 * c_out as f64 * (b * plane) as f64 * k as f64;
        println!(
            "  loop TILE={tile:<3} ({} iters): {:>8.1} ms  {:.2} TFLOP/s",
            b.div_ceil(tile), e * 1e3, flop / e / 1e12
        );
    }
}

fn im2col(x: &[f32], b: usize, c_in: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let h_out = (h + 2 - 3) / 2 + 1;
    let w_out = (w + 2 - 3) / 2 + 1;
    let plane = h_out * w_out;
    let k = c_in * 9;
    let mut cols = vec![0.0f32; b * plane * k];
    cols.par_chunks_mut(k).enumerate().for_each(|(col_idx, col)| {
        let ib = col_idx / plane;
        let rem = col_idx % plane;
        let ho = rem / w_out;
        let wo = rem % w_out;
        for ic in 0..c_in {
            for kh in 0..3 {
                for kw in 0..3 {
                    let ih = (ho * 2 + kh) as isize - 1;
                    let iw = (wo * 2 + kw) as isize - 1;
                    col[ic * 9 + kh * 3 + kw] = if ih < 0 || ih >= h as isize || iw < 0 || iw >= w as isize {
                        0.0
                    } else {
                        x[((ib * c_in + ic) * h + ih as usize) * w + iw as usize]
                    };
                }
            }
        }
    });
    (cols, h_out, w_out)
}

fn main() {
    println!("rayon threads: {}", rayon::current_num_threads());
    if std::env::args().any(|a| a == "--loop") {
        loop_probe();
        return;
    }
    println!("\n-- CUDA-shaped conv gemm (per TILE=8 batch of chunks) --");
    bench(240, 4320, 1600, 5);   // c2, TILE=8
    bench(480, 4320, 1600, 5);   // c2, TILE=16
    bench(960, 4320, 1600, 5);   // c2, TILE=32
    bench(2160, 4320, 1600, 3);  // c2, TILE=72

    println!("\n-- conv1 (k=27) / conv3 (k=1200) --");
    bench(240, 27, 1600, 5);
    bench(960, 27, 1600, 5);
    bench(240, 1200, 400, 5);
    bench(960, 1200, 400, 5);

    println!("\n-- transformer shapes (180s: s=2292; 15s: s=210) --");
    bench(2292, 1024, 1024, 5);
    bench(2292, 1024, 4096, 5);
    bench(2292, 4096, 1024, 5);
    bench(210, 1024, 1024, 5);
    bench(210, 1024, 4096, 5);

    println!("\n-- ffm shapes (windowed ffn: m = window count x 1250) --");
    bench(1250, 1024, 4096, 5);
    bench(1250, 4096, 1024, 5);

    println!("\n-- im2col write cost --");
    bench_im2col(8, 480, 32, 50);
    bench_im2col(32, 480, 32, 50);
}
