//! Print one generated batch shader: `dump_batch gemv|gqa|extract|silu`.
use qwen3_asr_wgpu::shaders;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let src = match a.get(1).map(|s| s.as_str()).unwrap_or("gemv") {
        "gemv" => shaders::gemv_batch_slots(2048, 2048, 3, false),
        "gqa" => shaders::gqa_decode_single_slots(16, 8, 128, 256, 1024),
        "extract" => shaders::qkv_extract_slots(16, 8, 128),
        "silu" => shaders::silu_mul_split_slots(6144),
        "norm" => shaders::gemv_norm_batch_slots(2048, 2048, 2, false, 2048, 1024, 1e-6),
        other => panic!("unknown {other}"),
    };
    println!("{src}");
}
