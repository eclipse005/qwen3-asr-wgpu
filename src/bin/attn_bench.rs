//! Microbenchmark for decode attention's stage 1 — the `Q . K` score loop.
//!
//! The engine's own probes have narrowed this kernel to a plateau: the paired
//! kernel (half the KV bytes) is worth +1.0%, `QASR_GQA_PF` (loads one word
//! early) +0.9%, the cooperative form (4x fewer cache lines per instruction)
//! +1.3% and it fails the gate, while the row0 probe (the whole memory system
//! removed) is worth 69% and `gqa_p1_coal` (coalesced addresses, 16 loads per
//! lane) 18%.  Two readings survive that evidence:
//!
//!   (a) **not enough warps** -- the kernel is latency-bound and the grid is
//!       too thin to hide it.  Then adding *redundant* workgroups in `wgid.z`
//!       raises the per-replica rate, and so does more positions per thread.
//!   (b) **not enough memory-level parallelism per thread** -- each lane walks
//!       its own 256 B row through 16 loads that the compiler may well keep
//!       serial.  Then the fix is to issue more of them before consuming any,
//!       which is a pure `--depth` change and keeps every FMA in order.
//!
//! Both are cheap to distinguish and neither is a guess.  `depth` issues `n`
//! K-row loads before consuming any of them and then replays the *identical*
//! add sequence (same operands, same order), so every variant here stays
//! bit-identical by construction; `pin` additionally holds 2 rows per thread in
//! flight; `mult` adds redundant workgroups in `wgid.z`; `stream` reads the
//! same byte count with fully coalesced addresses (the roof); `row0` pins every
//! row to row 0 (L1-resident ceiling).  `depth 0` is the shipped order, kv
//! load interleaved with its qa/qb broadcasts.
//!
//! Run from `target/release`.

use std::time::Instant;

use anyhow::Result;

use qwen3_asr_wgpu::gpu::Gpu;

const D2: usize = 64; // words per K/V row (d/2)
const D4: usize = 16; // vec4<u32> per K row
const NKVH: usize = 8;
const MAXSEQ: usize = 2688;
const CURLEN: usize = 2560;
const BS: usize = 256;

/// `dot = dot + (a.x*k.x + a.y*k.y)` -- the arithmetic the 12 transcripts pin
/// down, emitted identically in every variant.
fn add_line(h: &str, p: usize, j: usize, s: usize) -> String {
    format!(
        "            {h} = {h} + (a{p}_{j}_{s}.x * k{p}_{j}_{s}.x + a{p}_{j}_{s}.y * k{p}_{j}_{s}.y);\n"
    )
}

fn consume(pin: usize, j: usize) -> String {
    let mut s = String::new();
    for p in 0..pin {
        for sub in 0..4 {
            let w = ["x", "y", "z", "w"][sub];
            s.push_str(&format!(
                "            let k{p}_{j}_{sub} = unpack2x16float(kv{p}_{j}.{w});\n"
            ));
            s.push_str(&format!("            let a{p}_{j}_{sub} = unpack2x16float(qa.{w});\n"));
        }
        for sub in 0..4 {
            s.push_str(&add_line(&format!("da{p}"), p, j, sub));
        }
        for sub in 0..4 {
            let w = ["x", "y", "z", "w"][sub];
            s.push_str(&format!("            let b{p}_{j}_{sub} = unpack2x16float(qb.{w});\n"));
        }
        for sub in 0..4 {
            s.push_str(&add_line(&format!("db{p}"), p, j, sub));
        }
    }
    s
}

/// Fully unrolled stage-1 body.  `depth == 0` is the shipped interleave;
/// `depth == n` issues `n` K-row loads before consuming any of them;
/// `pf` reproduces `QASR_GQA_PF` exactly (the shipped default): the loop is
/// left rolled and `kv`/`qa`/`qb` hold one iteration's slack.
fn stage1(pin: usize, depth: usize, pf: bool, row0: bool, stream: bool) -> String {
    let mut s = String::new();
    let stride = BS * pin;
    if pin == 1 {
        s.push_str("    for (var t = lid.x; t < chunk_len; t = t + BS) {\n");
    } else {
        s.push_str(&format!(
            "    for (var t = lid.x; t + {BS}u < chunk_len; t = t + {stride}u) {{\n"
        ));
    }
    for p in 0..pin {
        let off = if p == 0 { String::new() } else { format!(" + {}u", p * BS) };
        s.push_str(&format!("        var da{p} = 0.0;\n        var db{p} = 0.0;\n"));
        if row0 {
            s.push_str(&format!("        let row4_{p} = 0u;\n"));
        } else {
            s.push_str(&format!(
                "        let row4_{p} = (kbase + (t_start + t{off}) * D2) >> 2u;\n"
            ));
        }
    }
    if stream {
        // Same byte count for the workgroup (chunk * 256 B), fully coalesced:
        // 16 instructions of 256 consecutive words each.
        s.push_str("        let wb = kbase + t_start * D2;\n");
        s.push_str("        {\n");
        for i in 0..D4 {
            s.push_str(&format!(
                "            let kv0_{i} = KC4[wb + {i}u * 256u + lid.x];\n"
            ));
        }
        for i in 0..D4 {
            s.push_str(&format!(
                "            da0 = da0 + f32(kv0_{i}.x & 0xFFFFu) + f32(kv0_{i}.y & 0xFFFFu);\n"
            ));
        }
        s.push_str("        }\n");
    } else if pf {
        // The shipped `QASR_GQA_PF` form, reproduced verbatim from
        // `shaders.rs:1458`: rolled loop, one iteration of slack in kp/qap/qbp.
        s.push_str(
            "        var kp = KC4[row4_0];\n\
             \x20       var qap = Q4[qa4];\n\
             \x20       var qbp = Q4[qb4];\n\
             \x20       for (var j4 = 0u; j4 < 16u; j4 = j4 + 1u) {\n\
             \x20           let kv = kp;\n\
             \x20           let qa = qap;\n\
             \x20           let qb = qbp;\n\
             \x20           let jn = min(j4 + 1u, 15u);\n\
             \x20           kp = KC4[row4_0 + jn];\n\
             \x20           qap = Q4[qa4 + jn];\n\
             \x20           qbp = Q4[qb4 + jn];\n",
        );
        for sub in 0..4 {
            let w = ["x", "y", "z", "w"][sub];
            s.push_str(&format!(
                "            let k{sub} = unpack2x16float(kv.{w});\n\
                 \x20           let a{sub} = unpack2x16float(qa.{w});\n"
            ));
        }
        for sub in 0..4 {
            s.push_str(&format!(
                "            da0 = da0 + (a{sub}.x * k{sub}.x + a{sub}.y * k{sub}.y);\n"
            ));
        }
        for sub in 0..4 {
            let w = ["x", "y", "z", "w"][sub];
            s.push_str(&format!("            let b{sub} = unpack2x16float(qb.{w});\n"));
        }
        for sub in 0..4 {
            s.push_str(&format!(
                "            db0 = db0 + (b{sub}.x * k{sub}.x + b{sub}.y * k{sub}.y);\n"
            ));
        }
        s.push_str("        }\n");
    } else if depth == 0 {
        for j in 0..D4 {
            s.push_str("        {\n");
            for p in 0..pin {
                s.push_str(&format!("            let kv{p}_{j} = KC4[row4_{p} + {j}u];\n"));
            }
            s.push_str(&format!("            let qa = Q4[qa4 + {j}u];\n"));
            s.push_str(&format!("            let qb = Q4[qb4 + {j}u];\n"));
            s.push_str(&consume(pin, j));
            s.push_str("        }\n");
        }
    } else {
        let mut j0 = 0usize;
        while j0 < D4 {
            let d = depth.min(D4 - j0);
            s.push_str("        {\n");
            for p in 0..pin {
                for k in 0..d {
                    let j = j0 + k;
                    s.push_str(&format!("            let kv{p}_{j} = KC4[row4_{p} + {j}u];\n"));
                }
            }
            for k in 0..d {
                let j = j0 + k;
                s.push_str("            {\n");
                s.push_str(&format!("            let qa = Q4[qa4 + {j}u];\n"));
                s.push_str(&format!("            let qb = Q4[qb4 + {j}u];\n"));
                s.push_str(&consume(pin, j));
                s.push_str("            }\n");
            }
            s.push_str("        }\n");
            j0 += d;
        }
    }
    for p in 0..pin {
        let off = if p == 0 { String::new() } else { format!(" + {}u", p * BS) };
        s.push_str(&format!("        sc_a[t{off}] = da{p} * cfg.scale;\n"));
        s.push_str(&format!("        sc_b[t{off}] = db{p} * cfg.scale;\n"));
    }
    s.push_str("    }\n");
    s
}

fn source(chunk: usize, pin: usize, depth: usize, pf: bool, row0: bool, stream: bool) -> String {
    format!(
        "\
struct Cfg {{ cur_len: u32, max_seq: u32, scale: f32, n_chunks: u32 }};

@group(0) @binding(0) var<storage, read>       Q4:   array<vec4<u32>>;
@group(0) @binding(1) var<storage, read>       KC4:  array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> Out:  array<f32>;
@group(0) @binding(3) var<uniform>             cfg:  Cfg;

const D2: u32 = 64u;
const CHUNK: u32 = {chunk}u;
const BS: u32 = 256u;

var<workgroup> sc_a: array<f32, {chunk}u>;
var<workgroup> sc_b: array<f32, {chunk}u>;

@compute @workgroup_size(256)
fn gqa_split_p1_pair(@builtin(workgroup_id) wgid: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>) {{
    let kh = wgid.x;
    let by = wgid.y;
    let qa4 = (kh * 2u) * D2 >> 2u;
    let qb4 = (kh * 2u + 1u) * D2 >> 2u;
    // Replica `wgid.z` reads its own 8-head slab, so `mult` buys genuine
    // parallel work over fresh bytes instead of L2 reuse of the same slab.
    let kbase = (kh + wgid.z * 8u) * cfg.max_seq * D2;
    let t_start = by * CHUNK;
    if (t_start >= cfg.cur_len) {{ return; }}
    let chunk_len = min(CHUNK, cfg.cur_len - t_start);

{body}
    // One broadcast load, so `Q4` stays in the layout for the `stream` variant
    // (which reads K with coalesced addresses only and never touches Q).
    let qmark = f32(Q4[0u].x & 1u);
    workgroupBarrier();
    // Parallel write-out.  A single-thread serial fold over `sc_a`/`sc_b` would
    // add ~5 us of latency per workgroup and swamp every variant equally.
    if (lid.x < chunk_len) {{
        let ob = ((kh * 16u + by) * 2048u) + wgid.z * 32768u;
        Out[ob + lid.x] = sc_a[lid.x] + qmark;
        Out[ob + 1024u + lid.x] = sc_b[lid.x];
    }}
}}
",
        chunk = chunk,
        body = stage1(pin, depth, pf, row0, stream),
    )
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let adapter = arg(&args, "--adapter");
    let gpu = pollster::block_on(Gpu::new(adapter.as_deref()))?;
    println!("adapter: {}", gpu.describe());

    let max_seq = MAXSEQ;
    // 32 head slabs so `mult` up to 4 can read four disjoint 5.2 MB regions.
    let k_words = 32 * max_seq * D2;
    let k_bytes = k_words * 16;

    let mut up = gpu.uploader();
    let kc = up.storage("kc", k_bytes as u64);
    let mut kd = vec![0u8; k_bytes];
    let mut st: u32 = 99991;
    for w in 0..k_words {
        for e in 0..8u32 {
            st = st.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = half::f16::from_f32(((st >> 9) as f32 / 8388608.0) - 0.5);
            let bits = v.to_bits().to_le_bytes();
            kd[(w * 8 + e as usize) * 2] = bits[0];
            kd[(w * 8 + e as usize) * 2 + 1] = bits[1];
        }
    }
    up.upload(&kc, &kd)?;
    let q = up.storage("q", (16 * D2 * 2) as u64);
    up.upload(&q, &vec![0x3cu8; 16 * D2 * 2])?;
    let out = up.storage("out", 524_288 * 4);
    let cfg = up.uniform("cfg", 16);
    up.finish()?;

    // Bytes stage 1 reads per layer: kv_heads * cur_len * 256 B per K row.
    let read_bytes = (NKVH * CURLEN * D4 * 16) as f64;

    let iters = 30usize;
    // (chunk, pin, depth); depth 99 means the shipped `pf` form.
    let scenarios: [(usize, usize, usize); 12] = [
        (256, 1, 99),
        (256, 1, 1),
        (256, 1, 2),
        (256, 1, 4),
        (256, 1, 8),
        (512, 1, 99),
        (512, 1, 4),
        (512, 1, 8),
        (1024, 1, 99),
        (1024, 1, 8),
        (512, 2, 8),
        (2048, 1, 8),
    ];
    let mults = [1u32, 2, 4];

    println!(
        "\ncur_len={CURLEN}  K bytes read/layer={:.2} MB  engine stage1 = 110.6 us/layer at chunk 256 \
         (28 layers)",
        read_bytes / 1e6
    );
    println!(
        "{:<7} {:>5} {:>4} {:>5} {:>5} {:>10} {:>9} {:>12}",
        "variant", "chunk", "pin", "depth", "mult", "ms/layer", "GB/s", "us/layer/rep"
    );
    println!("{}", "-".repeat(64));

    let variants: [(&str, bool, bool); 3] = [
        ("real", false, false),
        ("row0", true, false),
        ("stream", false, true),
    ];

    for (name, row0, stream) in variants {
        for &(chunk, pin, depth) in &scenarios {
            if chunk % (BS * pin) != 0 {
                continue;
            }
            if stream && (pin != 1 || chunk != 256 || (depth != 4 && depth != 99)) {
                continue;
            }
            if row0 && (pin != 1 || chunk != 256 || (depth != 4 && depth != 99)) {
                continue;
            }
            let src = source(chunk, pin, depth, depth == 99, row0, stream);
            let pipe = gpu.pipeline(name, &src, "gqa_split_p1_pair", None)?;
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: q.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: kc.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: cfg.as_entire_binding() },
                ],
            });
            let n_chunks = CURLEN.div_ceil(chunk);
            for &mult in &mults {
                let words: [u32; 4] = [
                    CURLEN as u32,
                    max_seq as u32,
                    0.125f32.to_bits(),
                    n_chunks as u32,
                ];
                gpu.queue.write_buffer(&cfg, 0, bytemuck::cast_slice(&words));
                let grid = (NKVH as u32, n_chunks as u32, mult);
                let run = || -> Result<f64> {
                    for _ in 0..3 {
                        let mut enc = gpu.device.create_command_encoder(&Default::default());
                        let mut cp = enc.begin_compute_pass(&Default::default());
                        cp.set_pipeline(&pipe);
                        cp.set_bind_group(0, &bg, &[]);
                        cp.dispatch_workgroups(grid.0, grid.1, grid.2);
                        drop(cp);
                        gpu.queue.submit([enc.finish()]);
                    }
                    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let mut enc = gpu.device.create_command_encoder(&Default::default());
                        let mut cp = enc.begin_compute_pass(&Default::default());
                        cp.set_pipeline(&pipe);
                        cp.set_bind_group(0, &bg, &[]);
                        cp.dispatch_workgroups(grid.0, grid.1, grid.2);
                        drop(cp);
                        gpu.queue.submit([enc.finish()]);
                    }
                    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
                    Ok(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64)
                };
                let ms = run()?;
                // `mult` replicas each read their own slab, so the useful
                // bytes-per-replica are unchanged; ms/mult is the per-replica
                // time and that is what scales with parallelism.
                let gbs = read_bytes / 1e9 / (ms / 1000.0);
                println!(
                    "{:<7} {:>5} {:>4} {:>5} {:>5} {:>8.3} {:>9.1} {:>10.1}",
                    name,
                    chunk,
                    pin,
                    depth,
                    mult,
                    ms,
                    gbs,
                    ms * 1000.0 / mult as f64
                );
            }
        }
        println!();
    }
    Ok(())
}
