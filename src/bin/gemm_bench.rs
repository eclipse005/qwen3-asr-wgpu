use std::time::Instant;

use anyhow::{bail, Result};

use qwen3_asr_wgpu::gpu::Gpu;

const BK: usize = 16;

/// `permb` stores the `BN`-row B tile with its local rows permuted: row
/// `r = 8*tx + j` lands in slot `16*j + tx` instead of slot `r`.
///
/// The compute read is `Bs[(slot) * PAD + q]`, and with PAD = 17 the per-lane
/// address of a warp's B load is `slot*17 + q`.  Unpermuted, `slot = 8*tx + j`,
/// so the 16 `tx` lanes stride `8*17 = 136` words, and `136 mod 32 = 8` means
/// every lane lands on one of four banks -- a 4-way conflict that costs four
/// LDS cycles per load.  The k-step issues 8 B loads and 64 FMAs, i.e. 32 LDS
/// cycles against 16 issue cycles for the FMAs at 4 warps/SM/clock: the shared
/// memory, not the FLOPs, sets the rate.
///
/// Permuted, the address is `(16*j + tx)*17 + q`, and `17*16 = 272 mod 32 = 16`
/// only shifts a whole 16-word block, so the 16 lanes land on banks
/// `17*tx mod 32` -- all 16 distinct -- while `j` moves in whole blocks.  The
/// values each thread reads, the order it accumulates them in and the FMA count
/// are all unchanged, so this is bit-identical by construction.
fn gemm_shader(tm: usize, tn: usize, double: bool, unroll_q: usize, permb: bool, v4: bool) -> String {
    let bm = 16 * tm;
    let bn = 16 * tn;
    let pad = BK + 1;
    let n_as = bm * BK / 256;
    let n_bs = bn * BK / 256;
    assert_eq!(bm * BK % 256, 0);
    assert_eq!(bn * BK % 256, 0);
    if v4 {
        assert_eq!(permb, true, "the vec4 B layout *is* the permutation");
        // The chunk store is written for `tm = tn = 8`: 512 chunks over 256
        // threads is two per thread, `c0` and `c0 + CSTEP`.  A smaller `tn`
        // would need its own inverse mapping, not a different constant.
        assert_eq!(n_as, 8, "v4 assumes two 4-row chunks per thread for A");
        assert_eq!(n_bs, 8, "v4 assumes two 4-row chunks per thread for B");
    }
    // Chunks (16 B = four rows at one k) per k-row, and the stride between a
    // thread's two chunks: `bm*bk/4` chunks over 256 threads.
    let (aq, bq) = (bm / 4, bn / 4);
    let cstep = bm * BK / 4 / 2;

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
         const PAD: u32 = {pad}u;\nconst TM: u32 = {tm}u;\nconst TN: u32 = {tn}u;\nconst AQ: u32 = {aq}u;\nconst BQ: u32 = {bq}u;\nconst CSTEP: u32 = {cstep}u;\n"
    ));
    if v4 {
        s.push_str(&format!("var<workgroup> A4: array<vec4<f32>, {}>;\n", bm * BK / 4));
        s.push_str(&format!("var<workgroup> B4: array<vec4<f32>, {}>;\n", bn * BK / 4));
    } else {
        s.push_str(&format!("var<workgroup> As: array<f32, {}>;\n", bm * pad));
        s.push_str(&format!("var<workgroup> Bs: array<f32, {}>;\n", bn * pad));
    }
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

    // ---- the vec4 (chunk) tile -------------------------------------------
    //
    // A chunk is 16 B = four rows at one k, and the tile is k-major:
    //
    //   A4[q * AQ + r/4]              rows 4*(r/4) .. +3 at k = q
    //   B4[q * BQ + pos(g, s)]        rows 8*g + 4*s .. +3 at k = q
    //
    // Thread `(tx, ty)` reads its eight A values as two chunks (rows 8*ty..+7)
    // and its eight B values as two chunks (rows 8*tx..+7).  A's address depends
    // only on `ty`, so a phase's eight lanes all want the same sixteen bytes --
    // a broadcast, no permutation needed.  B's sixteen `tx` lanes would stride
    // eight words and land on two bank groups per phase, which is what `pos`
    // fixes: it maps the eight `tx` values of a phase onto the eight groups.
    //
    // The store side is the inverse.  Thread `(tx, ty)` owns chunks
    // `c0 = ty*16 + tx` and `c0 + cstep`, i.e. `(k, rg) = (c0/AQ, c0%AQ)` and
    // `(k+8, rg)`.  Both halves of that pair are the *same four rows at two k*,
    // so one `odd` flag covers all eight of the thread's global words, and the
    // store is two whole chunks.  Each of the thread's global reads is one word
    // (two f16 k values) and the other k of each word is read by the thread that
    // owns it: the tile's words are read twice, which is free here because this
    // GEMM is compute-bound by two orders of magnitude on every shape it runs.
    let v4_setup = || -> String {
        format!(
            " let c0 = ty * 16u + tx;\n\
             \x20 let crg = c0 % {aq}u;\n let ck = c0 / {aq}u;\n\
             \x20 let cvodd = ((ck & 1u) == 1u);\n\
             \x20 let c2 = c0 + {cstep}u;\n",
            aq = aq,
            cstep = cstep
        )
    };
    // The two global reads of one chunk: four rows at k = `kx + ck + (0|8)`.
    let v4_read = |kx: &str, reg: usize, half: usize| -> String {
        let kk = format!("({kx} + ck + {}u)", half * 8);
        let mut t = String::new();
        for e in 0..4 {
            t.push_str(&format!(
                "   pfa{} = A[(m0 + 4u * crg + {e}u) * kk + {kk} / 2u];\n",
                reg + e
            ));
        }
        t
    };
    // The B row a chunk component holds; shared by the single-buffer and
    // prefetched store paths so they cannot drift apart.  (`v4_read_b` had
    // `4u * bog` without the `bos` term and only the double-buffered path was
    // wrong -- which is the one every variant but the control uses.)
    let b_row = |e: usize| format!("8u * bog + 4u * bos + {e}u");
    let v4_read_b = |kx: &str, reg: usize, half: usize| -> String {
        let kw = format!("({kx} + ck + {}u)", half * 8);
        let mut t = String::new();
        for e in 0..4 {
            t.push_str(&format!(
                "   pfb{} = W[(n0 + {row}) * kk + {kw} / 2u];\n",
                reg + e,
                row = b_row(e)
            ));
        }
        t
    };
    // B's chunk position `c2 % BQ = pos` inverts to (g, s) as
    // `g = ((pos>>4)<<3) | (pos&7)`, `s = (pos>>3)&1`.
    let bog_decl = format!(
        "   let bop = c2 % {bq}u;\n\
         \x20  let bog = ((bop >> 4u) << 3u) | (bop & 7u);\n\
         \x20  let bos = (bop >> 3u) & 1u;\n",
        bq = bq
    );

    let load_as = |kx: &str, dst: &str| -> String {
        let mut t = String::new();
        if v4 {
            // Two whole chunks: the same four rows at k = ck and k = ck+8.
            for half in 0..2 {
                let kw = format!("({kx} + ck + {}u)", half * 8);
                let mut comps = String::new();
                for e in 0..4 {
                    comps.push_str(&format!(
                        "\x20 halve(A[(m0 + 4u * crg + {e}u) * kk + {kw} / 2u], cvodd),"
                    ));
                }
                t.push_str(&format!(
                    "  A4[c0{}] = vec4<f32>({comps});\n",
                    if half == 0 { String::new() } else { " + CSTEP".into() }
                ));
            }
            return t;
        }
        for e in 0..n_as {
            t.push_str(&format!(
                "  {dst}[(ty + {}u) * PAD + tx] = halve(A[(m0 + ty + {}u) * kk + ({kx} + tx) / 2u], (tx & 1u) == 1u);\n",
                e * 16, e * 16
            ));
        }
        t
    };
    // Row `ty + 16e` of the B tile goes to slot `16*(v % tn) + v / tn` when
    // permuted -- the inverse of the compute read's `16*j + tx`.  Written in the
    // general form (a power of two, so it folds to a mask and a shift), the same
    // expression `shaders.rs` emits: a tn-hardcoded `(ty & 7)*16 + (ty >> 3)`
    // silently covered only half the tile at tn = 16 and made that variant fail
    // its own correctness check.
    let bslot = |e: usize| -> String {
        if permb {
            let v = format!("(ty + {}u)", e * 16);
            format!("(16u * ({v} % {tn}u) + {v} / {tn}u)")
        } else {
            format!("(ty + {}u)", e * 16)
        }
    };
    let load_bs = |kx: &str, dst: &str| -> String {
        let mut t = String::new();
        if v4 {
            // The same two chunks, but the rows come from the inverse of `pos`.
            for half in 0..2 {
                let kw = format!("({kx} + ck + {}u)", half * 8);
                let mut comps = String::new();
                for e in 0..4 {
                    comps.push_str(&format!(
                        "\x20 halve(W[(n0 + {row}) * kk + {kw} / 2u], cvodd),",
                        row = b_row(e)
                    ));
                }
                t.push_str(&format!(
                    "  B4[c0{}] = vec4<f32>({comps});\n",
                    if half == 0 { String::new() } else { " + CSTEP".into() }
                ));
            }
            return t;
        }
        for e in 0..n_bs {
            t.push_str(&format!(
                "  {dst}[{} * PAD + tx] = halve(W[(n0 + ty + {}u) * kk + ({kx} + tx) / 2u], (tx & 1u) == 1u);\n",
                bslot(e), e * 16
            ));
        }
        t
    };
    let pf_as = |kx: &str| -> String {
        if v4 {
            return v4_read(kx, 0, 0) + &v4_read(kx, 4, 1);
        }
        let mut t = String::new();
        for e in 0..n_as {
            t.push_str(&format!(
                "   pfa{e} = A[(m0 + ty + {}u) * kk + ({kx} + tx) / 2u];\n", e * 16
            ));
        }
        t
    };
    let pf_bs = |kx: &str| -> String {
        if v4 {
            return v4_read_b(kx, 0, 0) + &v4_read_b(kx, 4, 1);
        }
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
        if v4 {
            for half in 0..2 {
                let mut comps = String::new();
                for e in 0..4 {
                    comps.push_str(&format!(
                        "\x20 halve(pfa{}, cvodd),",
                        half * 4 + e
                    ));
                }
                t.push_str(&format!(
                    "   A4[c0{}] = vec4<f32>({comps});\n",
                    if half == 0 { String::new() } else { " + CSTEP".into() }
                ));
            }
            return t;
        }
        for e in 0..n_as {
            t.push_str(&format!(
                "   As[(ty + {}u) * PAD + tx] = halve(pfa{e}, (tx & 1u) == 1u);\n", e * 16
            ));
        }
        t
    };
    let store_bs = || {
        let mut t = String::new();
        if v4 {
            for half in 0..2 {
                let mut comps = String::new();
                for e in 0..4 {
                    comps.push_str(&format!(
                        "\x20 halve(pfb{}, cvodd),",
                        half * 4 + e
                    ));
                }
                t.push_str(&format!(
                    "   B4[c0{}] = vec4<f32>({comps});\n",
                    if half == 0 { String::new() } else { " + CSTEP".into() }
                ));
            }
            return t;
        }
        for e in 0..n_bs {
            t.push_str(&format!(
                "   Bs[{} * PAD + tx] = halve(pfb{e}, (tx & 1u) == 1u);\n", bslot(e)
            ));
        }
        t
    };
    let compute = || {
        let mut t = String::new();
        let step = |t: &mut String, q: &str, tag: &str| {
            if v4 {
                // Two chunks per operand: rows ty*8..+7 of A, tx*8..+7 of B.
                t.push_str(&format!("   let av{tag} = A4[{q}u * AQ + ty * 2u];\n"));
                t.push_str(&format!("   let aw{tag} = A4[{q}u * AQ + ty * 2u + 1u];\n"));
                t.push_str(&format!(
                    "   let bv{tag} = B4[{q}u * BQ + (tx & 7u) + 16u * (tx >> 3u)];\n"
                ));
                t.push_str(&format!(
                    "   let bw{tag} = B4[{q}u * BQ + (tx & 7u) + 8u + 16u * (tx >> 3u)];\n"
                ));
                for i in 0..tm {
                    let a = if i < 4 {
                        format!("av{tag}[{i}]")
                    } else {
                        format!("aw{tag}[{}]", i - 4)
                    };
                    for j in 0..tn {
                        let b = if j < 4 {
                            format!("bv{tag}[{j}]")
                        } else {
                            format!("bw{tag}[{}]", j - 4)
                        };
                        t.push_str(&format!("   c{i}{j} = c{i}{j} + {a} * {b};\n"));
                    }
                }
                return;
            }
            for i in 0..tm {
                t.push_str(&format!("   let a{i}_{q} = As[(ty * {tm}u + {i}u) * PAD + {q}];\n"));
            }
            for j in 0..tn {
                let idx = if permb {
                    format!("(16u * {j}u + tx)")
                } else {
                    format!("(tx * {tn}u + {j}u)")
                };
                t.push_str(&format!("   let b{j}_{q} = Bs[{idx} * PAD + {q}];\n"));
            }
            for i in 0..tm {
                for j in 0..tn {
                    t.push_str(&format!("   c{i}{j} = c{i}{j} + a{i}_{q} * b{j}_{q};\n"));
                }
            }
        };
        if unroll_q == 0 {
            if v4 {
                t.push_str("  var q: u32 = 0u;\n  loop {\n   if (q >= BK) { break; }\n");
                t.push_str("   let av = A4[q * AQ + ty * 2u];\n");
                t.push_str("   let aw = A4[q * AQ + ty * 2u + 1u];\n");
                t.push_str("   let bv = B4[q * BQ + (tx & 7u) + 16u * (tx >> 3u)];\n");
                t.push_str("   let bw = B4[q * BQ + (tx & 7u) + 8u + 16u * (tx >> 3u)];\n");
                for i in 0..tm {
                    let a = if i < 4 {
                        format!("av[{i}]")
                    } else {
                        format!("aw[{}]", i - 4)
                    };
                    for j in 0..tn {
                        let b = if j < 4 {
                            format!("bv[{j}]")
                        } else {
                            format!("bw[{}]", j - 4)
                        };
                        t.push_str(&format!("   c{i}{j} = c{i}{j} + {a} * {b};\n"));
                    }
                }
                t.push_str("   q = q + 1u;\n  }\n");
            } else {
            // The original runtime q-loop: one step per iteration.
            t.push_str("  var q: u32 = 0u;\n  loop {\n   if (q >= BK) { break; }\n");
            for i in 0..tm {
                t.push_str(&format!("   let a{i} = As[(ty * {tm}u + {i}u) * PAD + q];\n"));
            }
            for j in 0..tn {
                let idx = if permb {
                    format!("(16u * {j}u + tx)")
                } else {
                    format!("(tx * {tn}u + {j}u)")
                };
                t.push_str(&format!("   let b{j} = Bs[{idx} * PAD + q];\n"));
            }
            for i in 0..tm {
                for j in 0..tn {
                    t.push_str(&format!("   c{i}{j} = c{i}{j} + a{i} * b{j};\n"));
                }
            }
            t.push_str("   q = q + 1u;\n  }\n");
            }
        } else if unroll_q >= BK {
            for q in 0..BK {
                if v4 {
                    step(&mut t, &format!("{q}"), &format!("_{q}"));
                } else {
                    step(&mut t, &format!("{q}u"), &format!("_{q}"));
                }
            }
        } else {
            // Partial: `unroll_q` steps per iteration.
            t.push_str(&format!(
                "  var q0: u32 = 0u;\n  loop {{\n   if (q0 >= BK) {{ break; }}\n"
            ));
            for u in 0..unroll_q {
                if v4 {
                    let q = format!("(q0 + {u}u)");
                    t.push_str(&format!("   let av{u} = A4[{q} * AQ + ty * 2u];\n"));
                    t.push_str(&format!("   let aw{u} = A4[{q} * AQ + ty * 2u + 1u];\n"));
                    t.push_str(&format!(
                        "   let bv{u} = B4[{q} * BQ + (tx & 7u) + 16u * (tx >> 3u)];\n"
                    ));
                    t.push_str(&format!(
                        "   let bw{u} = B4[{q} * BQ + (tx & 7u) + 8u + 16u * (tx >> 3u)];\n"
                    ));
                    for i in 0..tm {
                        let a = if i < 4 {
                            format!("av{u}[{i}]")
                        } else {
                            format!("aw{u}[{}]", i - 4)
                        };
                        for j in 0..tn {
                            let b = if j < 4 {
                                format!("bv{u}[{j}]")
                            } else {
                                format!("bw{u}[{}]", j - 4)
                            };
                            t.push_str(&format!("   c{i}{j} = c{i}{j} + {a} * {b};\n"));
                        }
                    }
                    continue;
                }
                for i in 0..tm {
                    t.push_str(&format!(
                        "   let a{i}_{u} = As[(ty * {tm}u + {i}u) * PAD + (q0 + {u}u)];\n"
                    ));
                }
                for j in 0..tn {
                    let idx = if permb {
                        format!("(16u * {j}u + tx)")
                    } else {
                        format!("(tx * {tn}u + {j}u)")
                    };
                    t.push_str(&format!(
                        "   let b{j}_{u} = Bs[{idx} * PAD + (q0 + {u}u)];\n"
                    ));
                }
                for i in 0..tm {
                    for j in 0..tn {
                        t.push_str(&format!(
                            "   c{i}{j} = c{i}{j} + a{i}_{u} * b{j}_{u};\n"
                        ));
                    }
                }
            }
            t.push_str(&format!("   q0 = q0 + {unroll_q}u;\n  }}\n"));
        }
        t
    };

    // The chunk index/parity the whole store path is written in terms of.
    if v4 {
        s.push_str(&v4_setup());
        s.push_str(&bog_decl);
    }

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
    unroll_q: usize,
    permb: bool,
    v4: bool,
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

    let src = gemm_shader(tm, tn, double, unroll_q, permb, v4);
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
    if let Some(spec) = arg(&args, "--dump") {
        // `--dump TMD,TND,UQ,PB` prints the generated WGSL instead of running.
        let spec: Vec<&str> = spec.split(',').collect();
        let tm: usize = spec[0].parse()?;
        let tn: usize = spec[1].parse()?;
        let uq: usize = spec[2].parse()?;
        let pb: bool = spec[3] == "1";
        let v4: bool = spec[4] == "1";
        print!("{}", gemm_shader(tm, tn, true, uq, pb, v4));
        return Ok(());
    }
    let gpu = pollster::block_on(Gpu::new(adapter.as_deref()))?;
    println!("adapter: {}", gpu.describe());

    let variants: [(usize, usize, bool, usize, bool, bool); 8] = [
        (8, 8, true, 0, true, false),
        (8, 8, true, 4, true, false),
        (8, 8, true, 4, false, false),
        (8, 8, true, 0, true, true),
        (8, 8, true, 4, true, true),
        (8, 8, true, 8, true, true),
        (8, 8, true, 16, true, true),
        (8, 8, false, 4, true, true),
    ];

    println!("-- correctness (m=256, n=256, k=256) --");
    let mut seed = 42u32;
    for (tm, tn, dbl, uq, pb, v4) in variants {
        let b = setup(&gpu, tm, tn, dbl, uq, pb, v4, 256, 256, 256, &mut seed)?;
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        b.dispatch_n(&mut enc, 1);
        gpu.queue.submit([enc.finish()]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
        let (max_abs, max_rel) = b.verify()?;
        let ok = max_rel < 0.02 && max_abs < 0.2;
        println!(
            "  {tm}x{tn} dbl={dbl} uq={uq} pb={pb} v4={v4}: max|Δ|={max_abs:.4} max_rel={max_rel:.4} {}",
            if ok { "OK" } else { "FAIL" }
        );
        if !ok {
            bail!("variant {tm}x{tn} dbl={dbl} uq={uq} pb={pb} v4={v4} failed correctness");
        }
    }

    // (name, m, k, n).  The last four are the encoder transformer's exact shapes,
    // which the `QASR_ENC_DUP` ablation found running at 1.17-1.65 TFLOP/s in the
    // engine; the first five are the decoder prefill's, at ~1.9.  Same question
    // as `attn_bench`: is the encoder's rate a property of the shape, or of the
    // engine's dispatch of it?
    //
    // `m2304 k1024 n4096` is a multiple of 256 in both m and n, so it is the one
    // row where a BM/BN = 256 variant is not paying padding for its tile and the
    // tile *shapes* can be compared without that confound.
    let shapes: [(&str, usize, usize, usize); 10] = [
        ("gate m384 k1024 n4096", 384, 1024, 4096),
        ("gate m2304 k1024 n4096", 2304, 1024, 4096),
        ("gu   m384 k1024 n6144", 384, 1024, 6144),
        ("dp   m384 k3072 n1024", 384, 3072, 1024),
        ("o    m384 k2048 n1024", 384, 2048, 1024),
        ("enc qkv m1170 k896 n2688", 1170, 896, 2688),
        ("enc o   m1170 k896 n896", 1170, 896, 896),
        ("enc fc1 m1170 k896 n3584", 1170, 896, 3584),
        ("enc fc2 m1170 k3584 n896", 1170, 3584, 896),
        ("conv m512 k4320 n480", 512, 4320, 480),
    ];
    let iters = 20u32;
    println!("\n-- sweep, TFLOP/s (1-encoder timing) --");
    print!("{:<26}", "shape");
    for (tm, tn, dbl, uq, pb, v4) in variants {
        print!(
            " {:>10}",
            format!(
                "{tm}x{tn}{}{}{}{}",
                if dbl { "D" } else { "" },
                if uq > 0 { format!("U{uq}") } else { String::new() },
                if pb { "P" } else { "" },
                if v4 { "V" } else { "" }
            )
        );
    }
    println!();
    for (name, m, k, n) in shapes {
        print!("{:<26}", name);
        for (tm, tn, dbl, uq, pb, v4) in variants {
            let mut s2 = seed.wrapping_add(1);
            seed = s2;
            let b = setup(&gpu, tm, tn, dbl, uq, pb, v4, m, n, k, &mut s2)?;
            let ms = b.time(iters)?;
            let tf = 2.0 * m as f64 * n as f64 * k as f64 / 1e12 / (ms / 1000.0);
            print!(" {:>10.2}", tf);
        }
        println!();
    }
    Ok(())
}
