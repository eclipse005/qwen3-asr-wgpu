//! Verify the W8 kernels against the host dequant + f32 reference, value by
//! value.  This is the M3 equivalent of `gemm_bench`'s CPU matmul check: a
//! fixture-level drift says *that* something is wrong; this says *which kernel*
//! and *how*.
//!
//! Usage: `w8_probe [which]` — `gemv`, `norm`, `gemm`, or all (default).

use qwen3_asr_wgpu::{gpu::Gpu, shaders};

/// Deterministic pseudo-random fill (an LCG, so runs are comparable).
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn i8(&mut self) -> i8 {
        (self.next_u32() % 256) as u8 as i8
    }
    fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.next_u32() as f32 / u32::MAX as f32) * (hi - lo)
    }
}

fn main() -> anyhow::Result<()> {
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    let gpu = pollster::block_on(Gpu::new(Some("vulkan")))?;
    let subgroup = true;
    if which == "gemv" || which == "all" {
        probe_gemv(&gpu, subgroup)?;
    }
    if which == "norm" || which == "all" {
        probe_norm(&gpu, subgroup)?;
    }
    if which == "gemm" || which == "all" {
        probe_gemm(&gpu)?;
    }
    Ok(())
}

fn probe_gemv(gpu: &Gpu, subgroup: bool) -> anyhow::Result<()> {
    let (rows, k) = (64usize, 1024usize);
    let mut rng = Lcg(0xDEB);
    let q: Vec<i8> = (0..rows * k).map(|_| rng.i8()).collect();
    let scale: Vec<f32> = (0..rows).map(|_| rng.f32_in(0.004, 0.03)).collect();
    let x: Vec<half::f16> = (0..k).map(|_| half::f16::from_f32(rng.f32_in(-1.0, 1.0))).collect();

    // Reference: the M2 host dequant, then an f32 dot.
    let mut wf = vec![0.0f32; rows * k];
    for r in 0..rows {
        for c in 0..k {
            wf[r * k + c] = half::f16::from_f32(q[r * k + c] as f32 * scale[r]).to_f32();
        }
    }
    let xf: Vec<f32> = x.iter().map(|h| h.to_f32()).collect();
    let mut want = vec![0.0f64; rows];
    for r in 0..rows {
        let mut acc = 0.0f64;
        for c in 0..k {
            acc += (wf[r * k + c] * xf[c]) as f64;
        }
        want[r] = acc;
    }

    let pipe = gpu.pipeline("probe", &shaders::gemv_w8(rows, k, false, subgroup, 8), "gemv", None)?;
    let w_buf = gpu.storage("w", (rows * k) as u64);
    let x_buf = gpu.storage("x", (k * 2) as u64);
    let y_buf = gpu.storage("y", (rows * 2) as u64);
    let s_buf = gpu.storage("s", (rows * 4) as u64);
    let qb: Vec<u8> = q.iter().map(|&b| b as u8).collect();
    gpu.upload(&w_buf, &qb);
    gpu.upload(&x_buf, &weights::words_bytes(&x));
    let mut sb = Vec::with_capacity(rows * 4);
    for s in &scale {
        sb.extend_from_slice(&s.to_le_bytes());
    }
    gpu.upload(&s_buf, &sb);
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            e(0, &w_buf), e(1, &x_buf), e(2, &y_buf), e(3, &s_buf),
        ],
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&pipe);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups((rows / 8) as u32, 1, 1);
    }
    gpu.queue.submit([enc.finish()]);
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
    let got = gpu.readback(&y_buf, (rows * 2) as u64)?;
    let mut errs: Vec<(f64, usize)> = Vec::new();
    for r in 0..rows {
        let g = half::f16::from_le_bytes([got[r * 2], got[r * 2 + 1]]).to_f32() as f64;
        let d = ((g - want[r]) / want[r].abs().max(1e-9)).abs();
        errs.push((d, r));
    }
    errs.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (d, r) in errs.iter().take(3) {
        let g = half::f16::from_le_bytes([got[r * 2], got[r * 2 + 1]]).to_f32();
        println!("gemv row {r}: got {g:.6} want {:.6} rel {d:.2e}", want[*r]);
    }
    let worst = errs[0].0;
    println!("gemv_w8 worst rel err: {worst:.3e} — {}", ok(worst));
    Ok(())
}

fn e(b: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding: b, resource: buf.as_entire_binding() }
}

fn ok(worst: f64) -> &'static str {
    if worst < 2e-3 { "OK" } else { "WRONG" }
}

fn probe_norm(gpu: &Gpu, subgroup: bool) -> anyhow::Result<()> {
    let (rows, k) = (64usize, 1024usize);
    let eps = 1e-6f32;
    let mut rng = Lcg(0xBEE);
    let q: Vec<i8> = (0..rows * k).map(|_| rng.i8()).collect();
    let scale: Vec<f32> = (0..rows).map(|_| rng.f32_in(0.004, 0.03)).collect();
    let x: Vec<half::f16> = (0..k).map(|_| half::f16::from_f32(rng.f32_in(-1.0, 1.0))).collect();
    let nw: Vec<half::f16> = (0..k).map(|_| half::f16::from_f32(rng.f32_in(0.5, 1.5))).collect();

    // Reference: rms_norm's own math (f16 rounding of the normalized row, as
    // `norm_word` does), then the same f32 dot as `probe_gemv`.
    let sumsq: f32 = x.iter().map(|h| h.to_f32() * h.to_f32()).sum();
    let inv = 1.0 / (sumsq / k as f32 + eps).sqrt();
    let xn: Vec<f32> = x
        .iter()
        .zip(nw.iter())
        .map(|(h, w)| half::f16::from_f32(h.to_f32() * inv * w.to_f32()).to_f32())
        .collect();
    let mut want = vec![0.0f64; rows];
    for r in 0..rows {
        let mut acc = 0.0f64;
        for c in 0..k {
            let w = half::f16::from_f32(q[r * k + c] as f32 * scale[r]).to_f32();
            acc += (w * xn[c]) as f64;
        }
        want[r] = acc;
    }

    let bs = 1024usize;
    let pipe = gpu.pipeline(
        "probe",
        &shaders::gemv_norm_w8(rows, k, false, subgroup, k, bs, eps),
        "gemv",
        None,
    )?;
    let w_buf = gpu.storage("w", (rows * k) as u64);
    let x_buf = gpu.storage("x", (k * 2) as u64);
    let y_buf = gpu.storage("y", (rows * 2) as u64);
    let n_buf = gpu.storage("n", (k * 2) as u64);
    let s_buf = gpu.storage("s", (rows * 4) as u64);
    let qb: Vec<u8> = q.iter().map(|&b| b as u8).collect();
    gpu.upload(&w_buf, &qb);
    gpu.upload(&x_buf, &weights::words_bytes(&x));
    gpu.upload(&n_buf, &weights::words_bytes(&nw));
    let mut sb = Vec::with_capacity(rows * 4);
    for s in &scale {
        sb.extend_from_slice(&s.to_le_bytes());
    }
    gpu.upload(&s_buf, &sb);
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[e(0, &w_buf), e(1, &x_buf), e(2, &y_buf), e(3, &n_buf), e(4, &s_buf)],
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&pipe);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups((rows / 8) as u32, 1, 1);
    }
    gpu.queue.submit([enc.finish()]);
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
    let got = gpu.readback(&y_buf, (rows * 2) as u64)?;
    let mut worst = 0.0f64;
    for r in 0..rows {
        let g = half::f16::from_le_bytes([got[r * 2], got[r * 2 + 1]]).to_f32() as f64;
        let d = ((g - want[r]) / want[r].abs().max(1e-9)).abs();
        worst = worst.max(d);
        if r < 4 {
            println!("norm row {r}: got {g:.6} want {:.6} rel {d:.2e}", want[r]);
        }
    }
    println!("gemv_norm_w8 worst rel err: {worst:.3e} — {}", ok(worst));
    Ok(())
}

fn probe_gemm(gpu: &Gpu) -> anyhow::Result<()> {
    let (m, n, k) = (128usize, 128usize, 1024usize);
    let f16arm = std::env::var("W8_PROBE_F16").is_ok();
    let spike = std::env::var("W8_PROBE_SPIKE").is_ok();
    let mut rng = Lcg(0xFEED);
    let q: Vec<i8> = (0..n * k).map(|_| rng.i8()).collect();
    let scale: Vec<f32> = (0..n).map(|_| rng.f32_in(0.004, 0.03)).collect();
    let a: Vec<half::f16> = (0..m * k).map(|_| half::f16::from_f32(rng.f32_in(-1.0, 1.0))).collect();
    // Spike mode: unit weights at a few (n, k) positions, scale 1 — any
    // row/byte misalignment shows up as a spike in the wrong output.
    let spikes: Vec<(usize, usize)> = vec![(0, 0), (1, 1), (2, 5), (3, 17), (4, 259), (5, 1000), (100, 512), (127, 1023)];
    let mut q = q;
    let mut scale = scale;
    if spike {
        for v in q.iter_mut() { *v = 0; }
        for v in scale.iter_mut() { *v = 1.0; }
        for (ni, ki) in &spikes {
            q[ni * k + ki] = 127;
        }
    }
    // f16 arm: the "weights" are the dequantized f16 themselves.
    let wf16: Vec<half::f16> = q
        .iter()
        .enumerate()
        .map(|(i, &b)| half::f16::from_f32(b as f32 * scale[i / k]))
        .collect();

    let mut want = vec![0.0f64; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f64;
            for c in 0..k {
                let w = half::f16::from_f32(q[ni * k + c] as f32 * scale[ni]).to_f32();
                acc += (a[mi * k + c].to_f32() * w) as f64;
            }
            want[mi * n + ni] = acc;
        }
    }

    let (pipe, wsrc): (wgpu::ComputePipeline, Vec<u8>) = if f16arm {
        (
            gpu.pipeline("probe", &shaders::prefill_gemm(false, false), "gemm", None)?,
            weights::words_bytes(&wf16),
        )
    } else {
        let qb: Vec<u8> = q.iter().map(|&b| b as u8).collect();
        (
            gpu.pipeline("probe", &shaders::prefill_gemm_w8(false), "gemm", None)?,
            qb,
        )
    };
    let a_buf = gpu.storage("a", (m * k * 2) as u64);
    let w_buf = gpu.storage("w", if f16arm { (n * k * 2) as u64 } else { (n * k) as u64 });
    let c_buf = gpu.storage("c", (m * n * 2) as u64);
    let s_buf = gpu.storage("s", (n * 4) as u64);
    gpu.upload(&a_buf, &weights::words_bytes(&a));
    gpu.upload(&w_buf, &wsrc);
    let mut sb = Vec::with_capacity(n * 4);
    for s in &scale {
        sb.extend_from_slice(&s.to_le_bytes());
    }
    if !f16arm {
        gpu.upload(&s_buf, &sb);
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct GDims {
        m: u32, n: u32, k: u32, ldc: u32, bsa: u32, bsb: u32, bsc: u32, beta: u32, row0: u32, lda: u32,
    }
    let gd = GDims {
        m: m as u32, n: n as u32, k: k as u32,
        // ldc/lda are in f16 ELEMENTS; the kernel halves them into words.
        ldc: n as u32,
        bsa: 0, bsb: 0, bsc: 0, beta: 0, row0: 0,
        lda: k as u32,
    };
    let u_gd = gpu.uniform("gd", 256);
    gpu.upload(&u_gd, bytemuck::bytes_of(&gd));
    let mut entries = vec![
        e(0, &a_buf), e(1, &w_buf), e(2, &c_buf),
        wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &u_gd, offset: 0, size: std::num::NonZeroU64::new(64),
            }),
        },
    ];
    if !f16arm {
        entries.push(e(4, &s_buf));
    }
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&pipe);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups((n / 128) as u32, (m / 128) as u32, 1);
    }
    gpu.queue.submit([enc.finish()]);
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
    let got = gpu.readback(&c_buf, (m * n * 2) as u64)?;
    let mut worst = 0.0f64;
    let mut worst_at = (0, 0);
    for mi in 0..m {
        for ni in 0..n {
            let off = 2 * (mi * n + ni);
            let g = half::f16::from_le_bytes([got[off], got[off + 1]]).to_f32() as f64;
            let d = ((g - want[mi * n + ni]) / want[mi * n + ni].abs().max(1e-9)).abs();
            if d > worst {
                worst = d;
                worst_at = (mi, ni);
            }
        }
    }
    // error pattern: per (m, n) block of 32, the fraction of entries off by >1e-2
    let mut bad = vec![0u32; (m / 16) * (n / 32)];
    let mut total_bad = 0u32;
    for mi in 0..m {
        for ni in 0..n {
            let off = 2 * (mi * n + ni);
            let g = half::f16::from_le_bytes([got[off], got[off + 1]]).to_f32() as f64;
            if ((g - want[mi * n + ni]) / want[mi * n + ni].abs().max(1e-9)).abs() > 1e-2 {
                bad[(mi / 16) * (n / 32) + ni / 32] += 1;
                total_bad += 1;
                if total_bad <= 12 {
                    println!("bad ({mi},{ni}): got {g:.6} want {:.6}", want[mi * n + ni]);
                }
            }
        }
    }
    println!("bad entries: {total_bad}/{}", m * n);
    for (bi, b) in bad.iter().enumerate() {
        print!("{b:4} ");
        if bi % (n / 32) == (n / 32) - 1 { println!(); }
    }
    let off = 2 * (worst_at.0 * n + worst_at.1);
    let g = half::f16::from_le_bytes([got[off], got[off + 1]]).to_f32();
    println!(
        "gemm_w8 worst ({}, {}): got {g:.6} want {:.6} rel {worst:.3e} — {}",
        worst_at.0, worst_at.1, want[worst_at.0 * n + worst_at.1], ok(worst)
    );
    Ok(())
}

use qwen3_asr_wgpu::weights;
