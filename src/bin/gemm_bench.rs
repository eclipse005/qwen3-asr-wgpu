use std::time::Instant;

use anyhow::{bail, Result};

use qwen3_asr_wgpu::gpu::Gpu;

const BK: usize = 16;

fn gemm_shader(tm: usize, tn: usize, double: bool, unroll_q: bool) -> String {
    let bm = 16 * tm;
    let bn = 16 * tn;
    let pad = BK + 1;
    let n_as = bm * BK / 256;
    let n_bs = bn * BK / 256;
    assert_eq!(bm * BK % 256, 0);
    assert_eq!(bn * BK % 256, 0);

    let mut s = String::new();
    s.push_str(
        "struct GDims { m: u32, n: u32, k: u32, _p: u32 };\n\
         @group(0) @binding(0) var<storage, read>       A: array<u32>;\n\
         @group(0) @binding(1) var<storage, read>       W: array<u32>;\n\
         @group(0) @binding(2) var<storage, read_write> C: array<u32>;\n\
         @group(0) @binding(3) var<uniform>             gd: GDims;\n",
    );
    s.push_str(&format!(
        "const BM: u32 = {bm}u;\nconst BN: u32 = {bn}u;\nconst BK: u32 = {BK}u;\n\
         const PAD: u32 = {pad}u;\nconst TM: u32 = {tm}u;\nconst TN: u32 = {tn}u;\n"
    ));
    s.push_str(&format!("var<workgroup> As: array<f32, {}>;\n", bm * pad));
    s.push_str(&format!("var<workgroup> Bs: array<f32, {}>;\n", bn * pad));
    s.push_str(
        "fn halve(w: u32, odd: bool) -> f32 {\n\
         \x20 let p = unpack2x16float(w);\n\
         \x20 return select(p.x, p.y, odd);\n\
         }\n",
    );
    s.push_str(
        "@compute @workgroup_size(16, 16)\n\
         fn gemm(@builtin(workgroup_id) wid: vec3<u32>,\n\
                 @builtin(local_invocation_id) lid: vec3<u32>) {\n\
         let tx = lid.x;\n let ty = lid.y;\n\
         let m0 = wid.y * BM;\n let n0 = wid.x * BN;\n let kk = gd.k / 2u;\n",
    );

    for i in 0..tm {
        for j in 0..tn {
            s.push_str(&format!("var c{i}{j} = 0.0;\n"));
        }
    }

    let load_as = |kx: &str, dst: &str| -> String {
        let mut t = String::new();
        for e in 0..n_as {
            t.push_str(&format!(
                "  {dst}[(ty + {}u) * PAD + tx] = halve(A[(m0 + ty + {}u) * kk + ({kx} + tx) / 2u], (tx & 1u) == 1u);\n",
                e * 16, e * 16
            ));
        }
        t
    };
    let load_bs = |kx: &str, dst: &str| -> String {
        let mut t = String::new();
        for e in 0..n_bs {
            t.push_str(&format!(
                "  {dst}[(ty + {}u) * PAD + tx] = halve(W[(n0 + ty + {}u) * kk + ({kx} + tx) / 2u], (tx & 1u) == 1u);\n",
                e * 16, e * 16
            ));
        }
        t
    };
    let pf_as = |kx: &str| -> String {
        let mut t = String::new();
        for e in 0..n_as {
            t.push_str(&format!(
                "   pfa{e} = A[(m0 + ty + {}u) * kk + ({kx} + tx) / 2u];\n", e * 16
            ));
        }
        t
    };
    let pf_bs = |kx: &str| -> String {
        let mut t = String::new();
        for e in 0..n_bs {
            t.push_str(&format!(
                "   pfb{e} = W[(n0 + ty + {}u) * kk + ({kx} + tx) / 2u];\n", e * 16
            ));
        }
        t
    };
    let pf_decl = || {
        let mut t = String::new();
        for e in 0..n_as {
            t.push_str(&format!("  var pfa{e}: u32 = 0u;\n"));
        }
        for e in 0..n_bs {
            t.push_str(&format!("  var pfb{e}: u32 = 0u;\n"));
        }
        t
    };
    let store_as = || {
        let mut t = String::new();
        for e in 0..n_as {
            t.push_str(&format!(
                "   As[(ty + {}u) * PAD + tx] = halve(pfa{e}, (tx & 1u) == 1u);\n", e * 16
            ));
        }
        t
    };
    let store_bs = || {
        let mut t = String::new();
        for e in 0..n_bs {
            t.push_str(&format!(
                "   Bs[(ty + {}u) * PAD + tx] = halve(pfb{e}, (tx & 1u) == 1u);\n", e * 16
            ));
        }
        t
    };
    let compute = || {
        let mut t = String::new();
        if !unroll_q {
            t.push_str("  var q: u32 = 0u;\n  loop {\n   if (q >= BK) { break; }\n");
            for i in 0..tm {
                t.push_str(&format!("   let a{i} = As[(ty * {tm}u + {i}u) * PAD + q];\n"));
            }
            for j in 0..tn {
                t.push_str(&format!("   let b{j} = Bs[(tx * {tn}u + {j}u) * PAD + q];\n"));
            }
            for i in 0..tm {
                for j in 0..tn {
                    t.push_str(&format!("   c{i}{j} = c{i}{j} + a{i} * b{j};\n"));
                }
            }
            t.push_str("   q = q + 1u;\n  }\n");
        } else {
            for q in 0..BK {
                for i in 0..tm {
                    t.push_str(&format!(
                        "   let a{i}_{q} = As[(ty * {tm}u + {i}u) * PAD + {q}u];\n"));
                }
                for j in 0..tn {
                    t.push_str(&format!(
                        "   let b{j}_{q} = Bs[(tx * {tn}u + {j}u) * PAD + {q}u];\n"));
                }
                for i in 0..tm {
                    for j in 0..tn {
                        t.push_str(&format!(
                            "   c{i}{j} = c{i}{j} + a{i}_{q} * b{j}_{q};\n"));
                    }
                }
            }
        }
        t
    };

    if !double {
        s.push_str(" var k0: u32 = 0u;\n loop {\n  if (k0 >= gd.k) { break; }\n");
        s.push_str(&load_as("k0", "As"));
        s.push_str(&load_bs("k0", "Bs"));
        s.push_str("  workgroupBarrier();\n");
        s.push_str(&compute());
        s.push_str("  workgroupBarrier();\n");
        s.push_str("  k0 = k0 + BK;\n }\n");
    } else {
        s.push_str(" var k0: u32 = 0u;\n");
        s.push_str(&load_as("k0", "As"));
        s.push_str(&load_bs("k0", "Bs"));
        s.push_str(" workgroupBarrier();\n");
        s.push_str(" loop {\n  if (k0 >= gd.k) { break; }\n");
        s.push_str("  let kn = k0 + BK;\n");
        s.push_str(&pf_decl());
        s.push_str("  if (kn < gd.k) {\n");
        s.push_str(&pf_as("kn"));
        s.push_str(&pf_bs("kn"));
        s.push_str("  }\n");
        s.push_str(&compute());
        s.push_str("  workgroupBarrier();\n");
        s.push_str("  if (kn < gd.k) {\n");
        s.push_str(&store_as());
        s.push_str(&store_bs());
        s.push_str("  }\n  workgroupBarrier();\n");
        s.push_str("  k0 = kn;\n }\n");
    }

    for i in 0..tm {
        s.push_str(&format!("let row{i} = m0 + ty * {tm}u + {i}u;\n"));
    }
    for i in 0..tm {
        for e in 0..tn / 2 {
            let je = 2 * e;
            let jo = je + 1;
            s.push_str(&format!(
                "let we{i}_{e} = (row{i} * gd.n + n0 + tx * {tn}u + {je}u) / 2u;\n"
            ));
            s.push_str(&format!(
                "C[we{i}_{e}] = pack2x16float(vec2<f32>(c{i}{je}, c{i}{jo}));\n"
            ));
        }
    }
    s.push_str("}\n");
    s
}

fn rand_words(count_words: usize, seed: &mut u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(count_words * 4);
    for _ in 0..count_words {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let lo = half::f16::from_f32(((*seed >> 8) as f32 / 16777216.0) * 2.0 - 1.0);
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let hi = half::f16::from_f32(((*seed >> 8) as f32 / 16777216.0) * 2.0 - 1.0);
        out.extend_from_slice(&(lo.to_bits() as u32 | ((hi.to_bits() as u32) << 16)).to_le_bytes());
    }
    out
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

struct Bench<'a> {
    gpu: &'a Gpu,
    pipe: wgpu::ComputePipeline,
    bg: wgpu::BindGroup,
    c_buf: wgpu::Buffer,
    grid: (u32, u32),
    m: usize,
    n: usize,
    k: usize,
    a_words: Vec<u8>,
    w_words: Vec<u8>,
}

fn setup<'a>(
    gpu: &'a Gpu,
    tm: usize,
    tn: usize,
    double: bool,
    unroll_q: bool,
    m: usize,
    n: usize,
    k: usize,
    seed: &mut u32,
) -> Result<Bench<'a>> {
    let bm = 16 * tm;
    let bn = 16 * tn;
    let mp = m.div_ceil(bm) * bm;
    let np = n.div_ceil(bn) * bn;
    let kk = k / 2;

    let mut up = gpu.uploader();
    let a_buf = up.storage("A", (mp * kk * 4) as u64);
    let w_buf = up.storage("W", (np * kk * 4) as u64);
    let c_buf = up.storage("C", (mp * np / 2 * 4) as u64);
    let dims = up.uniform("dims", 16);

    let a_words = rand_words(mp * kk, seed);
    let w_words = rand_words(np * kk, seed);
    up.upload(&a_buf, &a_words)?;
    up.upload(&w_buf, &w_words)?;
    let mut db = [0u8; 16];
    db[0..4].copy_from_slice(&(mp as u32).to_le_bytes());
    db[4..8].copy_from_slice(&(np as u32).to_le_bytes());
    db[8..12].copy_from_slice(&(k as u32).to_le_bytes());
    up.upload(&dims, &db)?;
    up.finish()?;

    let src = gemm_shader(tm, tn, double, unroll_q);
    let pipe = gpu.pipeline("gemm", &src, "gemm", None)?;
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gemm"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: c_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
        ],
    });
    Ok(Bench {
        gpu,
        pipe,
        bg,
        c_buf,
        grid: ((np / bn) as u32, (mp / bm) as u32),
        m: mp,
        n: np,
        k,
        a_words,
        w_words,
    })
}

impl Bench<'_> {
    fn dispatch(&self, cp: &mut wgpu::ComputePass) {
        cp.set_pipeline(&self.pipe);
        cp.set_bind_group(0, &self.bg, &[]);
        cp.dispatch_workgroups(self.grid.0, self.grid.1, 1);
    }

    fn dispatch_n(&self, enc: &mut wgpu::CommandEncoder, iters: u32) {
        let mut cp = enc.begin_compute_pass(&Default::default());
        for _ in 0..iters {
            self.dispatch(&mut cp);
        }
    }

    fn time(&self, iters: u32) -> Result<f64> {
        {
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            self.dispatch_n(&mut enc, iters);
            self.gpu.queue.submit([enc.finish()]);
            self.gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
        }
        let t0 = Instant::now();
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        self.dispatch_n(&mut enc, iters);
        self.gpu.queue.submit([enc.finish()]);
        self.gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
        Ok(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64)
    }

    fn verify(&self) -> Result<(f32, f32)> {
        let (rows, cols) = (self.m.min(256), self.n.min(256));
        let bytes = self.gpu.readback(&self.c_buf, (rows * cols / 2 * 4) as u64)?;
        let unpack = |w: &[u8]| -> Vec<f32> {
            w.chunks_exact(4)
                .flat_map(|w| {
                    let u = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
                    [
                        half::f16::from_bits((u & 0xffff) as u16).to_f32(),
                        half::f16::from_bits((u >> 16) as u16).to_f32(),
                    ]
                })
                .collect()
        };
        let c = unpack(&bytes);
        let a = unpack(&self.a_words);
        let wt = unpack(&self.w_words);

        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        for i in 0..rows {
            for j in 0..cols {
                let mut acc = 0f32;
                for l in 0..self.k {
                    acc += a[i * self.k + l] * wt[j * self.k + l];
                }
                let idx = i * self.n + j;
                let got = c[idx / 2 * 2 + (idx & 1)];
                let d = (acc - got).abs();
                max_abs = max_abs.max(d);
                max_rel = max_rel.max(d / acc.abs().max(1.0));
            }
        }
        eprintln!("debug: C[0..8] = {:?}  expected[0] ~ {:.4}", &c[..8.min(c.len())], {
            let mut acc = 0f32;
            for l in 0..self.k { acc += a[l] * wt[l]; }
            acc
        });
        Ok((max_abs, max_rel))
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let adapter = arg(&args, "--adapter");
    let gpu = pollster::block_on(Gpu::new(adapter.as_deref()))?;
    println!("adapter: {}", gpu.describe());

    let variants: [(usize, usize, bool, bool); 6] = [
        (8, 8, false, false),
        (8, 8, true, false),
        (8, 8, true, true),
        (8, 8, false, true),
        (8, 4, true, true),
        (4, 4, true, true),
    ];

    println!("-- correctness (m=256, n=256, k=256) --");
    let mut seed = 42u32;
    for (tm, tn, dbl, uq) in variants {
        let b = setup(&gpu, tm, tn, dbl, uq, 256, 256, 256, &mut seed)?;
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        b.dispatch_n(&mut enc, 1);
        gpu.queue.submit([enc.finish()]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
        let (max_abs, max_rel) = b.verify()?;
        let ok = max_rel < 0.02 && max_abs < 0.2;
        println!(
            "  {tm}x{tn} dbl={dbl} uq={uq}: max|Δ|={max_abs:.4} max_rel={max_rel:.4} {}",
            if ok { "OK" } else { "FAIL" }
        );
        if !ok {
            bail!("variant {tm}x{tn} dbl={dbl} failed correctness");
        }
    }

    // (name, m, k, n).  The last four are the encoder transformer's exact shapes,
    // which the `QASR_ENC_DUP` ablation found running at 1.17-1.65 TFLOP/s in the
    // engine; the first five are the decoder prefill's, at ~1.9.  Same question
    // as `attn_bench`: is the encoder's rate a property of the shape, or of the
    // engine's dispatch of it?
    let shapes: [(&str, usize, usize, usize); 9] = [
        ("gate m384 k1024 n4096", 384, 1024, 4096),
        ("gate m2304 k1024 n4096", 2304, 1024, 4096),
        ("gu   m384 k1024 n6144", 384, 1024, 6144),
        ("dp   m384 k3072 n1024", 384, 3072, 1024),
        ("o    m384 k2048 n1024", 384, 2048, 1024),
        ("enc qkv m1170 k896 n2688", 1170, 896, 2688),
        ("enc o   m1170 k896 n896", 1170, 896, 896),
        ("enc fc1 m1170 k896 n3584", 1170, 896, 3584),
        ("enc fc2 m1170 k3584 n896", 1170, 3584, 896),
    ];
    let iters = 20u32;
    println!("\n-- sweep, TFLOP/s (1-encoder timing) --");
    print!("{:<26}", "shape");
    for (tm, tn, dbl, uq) in variants {
        print!(
            " {:>10}",
            format!(
                "{tm}x{tn}{}{}",
                if dbl { "D" } else { "" },
                if uq { "U" } else { "" }
            )
        );
    }
    println!();
    for (name, m, k, n) in shapes {
        print!("{:<26}", name);
        for (tm, tn, dbl, uq) in variants {
            let mut s2 = seed.wrapping_add(1);
            seed = s2;
            let b = setup(&gpu, tm, tn, dbl, uq, m, n, k, &mut s2)?;
            let ms = b.time(iters)?;
            let tf = 2.0 * m as f64 * n as f64 * k as f64 / 1e12 / (ms / 1000.0);
            print!(" {:>10.2}", tf);
        }
        println!();
    }
    Ok(())
}
