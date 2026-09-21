//! Print any generated shader.  Added after the third time a generator change
//! needed its output read rather than guessed at -- `gemm_bench --dump` covers
//! the GEMM and nothing else did.
//!
//! Usage: `shader_dump <name> [args...]`, e.g.
//!   shader_dump softmax 32
//!   shader_dump softmax_tree 256 1024
//!   shader_dump gemm
//!   shader_dump list

use qwen3_asr_wgpu::shaders;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let name = a.get(1).map(|s| s.as_str()).unwrap_or("list");
    let validate = a.iter().any(|s| s == "--validate");
    let num = |i: usize| -> anyhow::Result<usize> {
        Ok(a.get(i)
            .ok_or_else(|| anyhow::anyhow!("missing argument {i}"))?
            .parse()?)
    };
    let src = match name {
        "softmax" => shaders::softmax_causal(num(2)?, sg()),
        "softmax_tree" => shaders::softmax_causal_tree(num(2)?, num(3)?, sg()),
        "gemm" => shaders::prefill_gemm(false, false),
        "gemm_causal" => shaders::prefill_gemm_causal(),
        "gemm_av" => shaders::prefill_gemm_causal_av(),
        "slab_stats" => shaders::slab_stats(num(2)?, num(3)?),
        // The shipped geometry unless told otherwise: Qwen3-ASR is
        // nqh 16 / nkvh 8 / hd 128 at both sizes.
        "gqa_split" => shaders::gqa_decode_split_p1(num(2)?, num(3)?, num(4)?, 256),
        "gqa_pair" => {
            shaders::gqa_decode_split_p1_pair(num(2)?, 256, 2, false, false, true, false, 1)
        }
        "gqa_pair_diag" => {
            // row0 / coal, the two access-pattern diagnostics
            shaders::gqa_decode_split_p1_pair(num(2)?, 256, 2, num(3)? == 1, false, false, num(4)? == 1, 1)
        }
        "gqa_single" => shaders::gqa_decode_single(num(2)?, num(3)?, num(4)?, num(5)?, 1024),
        "gqa_merge" => shaders::gqa_split_merge(num(2)?),
        "extract" => shaders::qkv_extract(num(2)?, num(3)?, num(4)?),
        "repeat_kv" => shaders::repeat_kv(num(2)?),
        "list" => {
            println!(
                "softmax <bs>\nsoftmax_tree <bp> <bt>\ngemm\ngemm_causal\ngemm_av\n\
                 slab_stats <bs> <t>\ngqa_split <nqh> <nkvh> <hd>\ngqa_pair <hd>\n\
                 gqa_pair_diag <hd> <row0> <coal>\ngqa_single <nqh> <nkvh> <hd> <bs>\n\
                 gqa_merge <hd>\nextract <nqh> <nkvh> <hd>\nrepeat_kv <nrep>"
            );
            return Ok(());
        }
        other => anyhow::bail!("unknown shader {other:?}; try `list`"),
    };
    if validate {
        // The engine's error chain stops at \Entry point X at Compute is invalid\;
        // creating the pipeline here surfaces wgpu\'s own parse/validation detail.
        let gpu = pollster::block_on(qwen3_asr_wgpu::gpu::Gpu::new(Some("vulkan")))?;
        println!("adapter: {}", gpu.describe());
        let entry = if name.starts_with("gqa") {
            if name == "gqa_merge" {
                "gqa_merge"
            } else if name == "gqa_single" {
                "gqa"
            } else if name == "gqa_pair" || name == "gqa_pair_diag" {
                "gqa_split_p1_pair"
            } else {
                "gqa_split_p1"
            }
        } else if name == "extract" {
            "qkv_extract"
        } else if name == "repeat_kv" {
            "repeat_kv"
        } else {
            "softmax"
        };
        match gpu.pipeline("probe", &src, entry, None) {
            Ok(_) => println!("pipeline OK"),
            Err(e) => println!("pipeline FAILED: {e}"),
        }
        return Ok(());
    }
    print!("{src}");
    Ok(())
}

/// The subgroup tail is on for this device (Pascal reports a fixed 32).
fn sg() -> bool {
    true
}
