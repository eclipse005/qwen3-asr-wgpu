# Design: tiled causal prefill attention (kills the s² scratch)

Status: **design, not implemented.**  This is the next structural change to the RTFx
work; it is the only known way to remove both the prefill attention traffic and the
~10-minute capacity wall in one move.

## The problem, measured

`decoder::prefill` materializes two `[nqh, mp, np]` f16 matrices per pass:

| | 180 s clip (s = 2307) | 15 min clip (s = 12065) |
|---|---|---|
| `scores` + `attn` | 2 × 170 MB | 2 × 4.7 GB |
| cost | 821 ms of the 1772 ms prefill (46 %), all of it traffic | **refused**: each buffer is past the 2047 MiB single-binding limit |

The ablation (`QASR_DUP=p_scores|p_softmax|p_av`) splits that 821 ms into
scores 295 / causal softmax 243 / AV 283 ms.  The causal skips already landed
(see HANDOFF) cut it to ~550 ms; the remaining cost is inherent to writing and
re-reading s².

## Design

Tile the **key** dimension.  With `T` = 1024 keys per tile (padded), the largest
scratch becomes `[nqh, mp, T]` — 9.4 MB at s = 2307, and *constant* in s:

    n_tiles = ceil(cur / T)
    for t in 0..n_tiles:                       # three passes over the tiles
        # pass A: per-tile row max, combined with a plain max (exact)
        scores_tile[t] = q × K[t]ᵀ             # GEMM, k = hd, n = T
        row_max = max(row_max, tile_max)       # max is order-free, no rounding
    for t in 0..n_tiles:
        # pass B: exp against the *global* max, tile sum, keep the exp'd tile
        scores_tile[t] = q × K[t]ᵀ             # recomputed (cheap: k = hd)
        e = exp(scores_tile[t] - row_max)
        row_sum += tile_sum(e)                 # grouped by tile, in tile order
        attn_tile[t] = e
    for t in 0..n_tiles:
        # pass C: accumulate the (unnormalized) AV per tile
        attn_flat += attn_tile[t] × V[t]       # GEMM with beta = accumulate
    attn_flat *= 1 / row_sum                   # small epilogue kernel

Why this is exact-ish rather than a rewrite of the math: pass A/B use max and sum
reductions that are *reassociated* relative to the flat kernel (the softmax's
block tree is fixed by the reference, so a per-tile sum will not be bit-identical)
— the same trade the split-K decode attention already makes.  **The transcript is
the gate**, not bit-equality.

Cost: one extra scores GEMM pass (k = hd only, ~5 ms per pass at 0.6B) and one
extra small epilogue, against ~3-4x less scratch traffic.  For 15-minute audio it
is the difference between "refused" and "runs".

## Where it plugs in

* `decoder::prefill` (the layer loop's steps 5-7): the three dispatches
  (`gemm`/`softmax`/`gemm_av`) become a tile loop; the buffers are allocated
  `[nqh, mp, T]` + `[nqh, mp]` (max/sum) instead of `[nqh, mp, np]`.
* `shaders::softmax_causal`: needs a tile form (row max / exp+sum / store) that
  reads a `cur`-strided tile, or three small new entry points.
* The AV GEMM already accumulates (`gemm_acc`/beta) — pass C is that with the
  k-range restricted to a tile, so it needs a `k0` offset in `GDims` (or a
  per-tile bind group with an offset `bsa`).
* **Gate it**: only take the tiled path when `s > 4096` (≈ 5.5 min of audio).
  The six fixtures and every FLEURS clip stay on the current path, so the
  alignment gate cannot move; the long path is verified by comparing its output
  against the current path on `5m.wav` / `6m.wav` / `10m.wav` (all three run
  today) and by finally running `15m.wav`.

## Why not flash-attention (online softmax)

Rescaling the accumulators per tile multiplies the running output by a changing
factor; that is a different (and larger) arithmetic change than re-grouping the
row sum, and it does not buy anything over the three-pass form here — the scores
GEMM is only ~5 ms of a ~1.8 s prefill, so recomputing it is free.
