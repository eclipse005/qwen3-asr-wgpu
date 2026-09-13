//! Minimal repro for the 1.7B embed-table problem: big buffer alloc + upload +
//! SSBO binding, isolating which stage fails on this driver.
//!
//! cargo run --release --manifest-path wgpu/Cargo.toml --bin buffer_probe -- --mb 593

use anyhow::Result;
use qwen3_asr_wgpu::gpu::Gpu;

const KERNEL: &str = "
@group(0) @binding(0) var<storage, read>       Table: array<u32>;
@group(0) @binding(1) var<storage, read_write> Out:   array<u32>;
@compute @workgroup_size(64)
fn copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    Out[gid.x] = Table[gid.x];
}
";

/// Exact replica of the decoder's `embed_lookup_single` kernel.
const EMBED_KERNEL: &str = "
struct Cfg { slot: u32, d2: u32, _a: u32, _b: u32 };

@group(0) @binding(0) var<storage, read>       Table: array<u32>;
@group(0) @binding(1) var<storage, read>       Ids:   array<i32>;
@group(0) @binding(2) var<storage, read_write> Out:   array<u32>;
@group(0) @binding(3) var<uniform>             cfg:   Cfg;

const BS: u32 = 1024u;

@compute @workgroup_size(1024)
fn embed(@builtin(local_invocation_id) lid: vec3<u32>) {
    let id = u32(Ids[cfg.slot]);
    let base = id * cfg.d2;
    for (var w = lid.x; w < cfg.d2; w = w + BS) {
        Out[w] = Table[base + w];
    }
}
";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mb: u64 = args
        .iter()
        .position(|a| a == "--mb")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(593);
    let bytes = mb * 1024 * 1024;
    let words = (bytes / 4) as usize;

    let gpu = pollster::block_on(Gpu::new(Some("nvidia")))?;
    println!("adapter: {}", gpu.describe());
    println!("features: {:?}
sub_group: {}", gpu.features, gpu.features.contains(wgpu::Features::SUBGROUP));
    println!("allocating {bytes} bytes ({words} words)...");
    let buf = gpu.storage("big", bytes);
    let out = gpu.storage("out", 4096);

    // 1. upload a recognizable pattern
    let pattern: Vec<u8> = (0..words as u32)
        .flat_map(|w| (w ^ 0xDEAD_BEEF).to_le_bytes())
        .collect();
    println!("uploading pattern (chunked)...");
    gpu.upload(&buf, &pattern);

    // 2. readback through COPY (no binding involved)
    let back = gpu.readback(&buf, 16)?;
    let w0 = u32::from_le_bytes([back[0], back[1], back[2], back[3]]);
    println!("readback word[0] = {w0:#010x} (expect 0xdeadbeef) -> {}",
        if w0 == 0xdead_beef { "OK" } else { "FAIL" });

    // 3. bind the WHOLE buffer as SSBO and copy the first words through a kernel
    let pipe = gpu.pipeline("copy", KERNEL, "copy", None)?;
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("copy"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
        ],
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&pipe);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(1, 1, 1);
    }
    gpu.queue.submit([enc.finish()]);
    let got = gpu.readback(&out, 16)?;
    let g0 = u32::from_le_bytes([got[0], got[1], got[2], got[3]]);
    println!("kernel read word[0] = {g0:#010x} -> {}",
        if g0 == 0xdead_beef { "OK" } else { "FAIL" });

    // 4. exact embed-kernel replica against the big table
    let tok = 11528u32;
    let d2 = mb_size_words(&bytes, &args);
    let row_bytes = d2 * 4;
    let table = gpu.storage("table", bytes);
    // fill the token's row so a correct kernel returns a known value
    let row_pattern: Vec<u8> = (0..d2 as u32)
        .flat_map(|w| (w ^ 0x1234_5678).to_le_bytes())
        .collect();
    gpu.upload(&table, &pattern); // reuse big pattern
    gpu.queue.write_buffer(&table, tok as u64 * row_bytes as u64, &row_pattern);
    let ids = gpu.storage("ids", 4);
    gpu.queue.write_buffer(&ids, 0, &tok.to_le_bytes());
    let out2 = gpu.storage("out2", row_bytes as u64);
    let cfgb = gpu.uniform("cfg", 16);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Cfg { slot: u32, d2: u32, a: u32, b: u32 }
    gpu.queue.write_buffer(&cfgb, 0, bytemuck::bytes_of(&Cfg { slot: 0, d2, a: 0, b: 0 }));

    let pipe = gpu.pipeline("embed", EMBED_KERNEL, "embed", None)?;
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("embed"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: table.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: ids.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out2.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: cfgb.as_entire_binding() },
        ],
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(&pipe);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(1, 1, 1);
    }
    gpu.queue.submit([enc.finish()]);
    let got = gpu.readback(&out2, row_bytes as u64)?;
    let g0 = u32::from_le_bytes([got[0], got[1], got[2], got[3]]);
    println!("embed replica word[0] = {g0:#010x} (expect 0x12345678) -> {}",
        if g0 == 0x1234_5678 { "OK" } else { "FAIL" });
    Ok(())
}

/// Row size in u32 words: replicate the decoder — `hs/2` words per row, with the
/// table holding `vocab * hs` f16.  Here: derive from `--d2` or default 1024.
fn mb_size_words(_bytes: &u64, args: &[String]) -> u32 {
    args.iter()
        .position(|a| a == "--d2")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024)
}
