//! Print the capabilities the batch path's design depends on.
//!
//! `binding_array` (one binding for N slots instead of 2N) needs
//! `BUFFER_BINDING_ARRAY` plus non-uniform indexing on this stack; the count of
//! storage buffers per stage is what caps the per-slot-binding design at b=3.
use anyhow::Result;
use qwen3_asr_wgpu::gpu::Gpu;

fn main() -> Result<()> {
    let gpu = pollster::block_on(Gpu::new(None))?;
    println!("adapter: {}", gpu.describe());
    let f = gpu.features;
    for (name, on) in [
        ("BUFFER_BINDING_ARRAY", f.contains(wgpu::Features::BUFFER_BINDING_ARRAY)),
        ("SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING",
         f.contains(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING)),
        ("STORAGE_RESOURCE_BINDING_ARRAY", f.contains(wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY)),
        ("SUBGROUP", f.contains(wgpu::Features::SUBGROUP)),
    ] {
        println!("  {name:<70} {on}");
    }
    let l = gpu.limits;
    println!("  max_storage_buffers_per_shader_stage        {}", l.max_storage_buffers_per_shader_stage);
    println!("  max_compute_workgroup_storage_size          {}", l.max_compute_workgroup_storage_size);
    println!("  max_bind_groups                             {}", l.max_bind_groups);
    Ok(())
}
