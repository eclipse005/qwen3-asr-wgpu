//! Pure-CPU memory-bandwidth probe (streaming read/write).
//!
//! The CPU audio encoder is weight-bandwidth bound: ~380 MB of f32 weights per
//! pass.  This measures what the machine can actually stream, so "how much of
//! encode time is irreducible weight traffic" has a real number behind it.
//!
//! ```text
//! cargo run --release --bin membw_probe
//! ```

use std::time::Instant;

fn best_of<F: FnMut() -> f64>(reps: usize, mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let v = f();
        if v < best {
            best = v;
        }
    }
    best
}

fn main() {
    let n = 64usize << 20; // 256 MB of f32
    let a: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
    let mut c: Vec<f32> = vec![0.0; n];

    // 1. triad: 3 streams (2 read + 1 write)
    let s = best_of(5, || {
        let t = Instant::now();
        for i in 0..n {
            c[i] = a[i] + b[i];
        }
        t.elapsed().as_secs_f64()
    });
    println!("add (2R+1W)      : {:>7.1} ms  {:>6.1} GB/s", s * 1e3, 3.0 * n as f64 * 4.0 / s / 1e9);

    // 2. pure read (sum)
    let s = best_of(5, || {
        let t = Instant::now();
        let mut acc = 0.0f32;
        for i in 0..n {
            acc += a[i];
        }
        std::hint::black_box(acc);
        t.elapsed().as_secs_f64()
    });
    println!("read-only (1R)   : {:>7.1} ms  {:>6.1} GB/s", s * 1e3, n as f64 * 4.0 / s / 1e9);

    // 3. pure write (fill)
    let s = best_of(5, || {
        let t = Instant::now();
        for i in 0..n {
            c[i] = 1.0;
        }
        t.elapsed().as_secs_f64()
    });
    println!("write-only (1W)  : {:>7.1} ms  {:>6.1} GB/s", s * 1e3, n as f64 * 4.0 / s / 1e9);

    // 4. rayon-parallel add (what the encoder's elementwise ops do)
    use rayon::prelude::*;
    let s = best_of(5, || {
        let t = Instant::now();
        c.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((cc, aa), bb)| {
            *cc = *aa + *bb;
        });
        t.elapsed().as_secs_f64()
    });
    println!("rayon add (2R+1W): {:>7.1} ms  {:>6.1} GB/s", s * 1e3, 3.0 * n as f64 * 4.0 / s / 1e9);

    // 5. f16 -> f32 convert of a 190 MB f16 weight image (the GPU upload path)
    let src: Vec<half::f16> = (0..n).map(|i| half::f16::from_f32((i % 13) as f32)).collect();
    let mut dst: Vec<f32> = vec![0.0; n];
    let s = best_of(5, || {
        let t = Instant::now();
        dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, v)| *d = v.to_f32());
        t.elapsed().as_secs_f64()
    });
    println!("f16->f32 convert  : {:>7.1} ms  {:>6.1} GB/s", s * 1e3, 6.0 * n as f64 / s / 1e9);

    println!("\nencoder weight images per pass:");
    println!("  f32: ~380 MB -> {:.2} s at 100 GB/s, {:.2} s at 200 GB/s", 0.380 / 0.100, 0.380 / 0.200);
    println!("  f16: ~190 MB -> {:.2} s at 100 GB/s, {:.2} s at 200 GB/s", 0.190 / 0.100, 0.190 / 0.200);
}
