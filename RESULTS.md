# Measured runs — full TPS log

Every number below is from a real run on GB10 hardware (DGX Spark, sm_121a,
~273 GB/s LPDDR5X), serving the 117B-class MoE described in the README
(256 experts, top-10, INT4 group-32 experts, bf16 activations) on vLLM
0.25.1 with CUDA graphs. Decode rates are server-reported completion tokens
over the first→last chunk window (`bench/e2e_tps.py`); prefill rates are
server-reported prompt tokens over wall time. GPU graphics clock was locked
to 2400 MHz for the 07-26 runs and 2200 MHz for the 07-27 runs (thermal
posture — see the note at the bottom).

## Single node, spec-off (2026-07-26, 800-token decode)

| configuration | coding c1 | prose c1 |
|---|---|---|
| Marlin WNA16 MoE (stock vLLM default) | 19.6 | 19.5 |
| stock Triton + this repo's tuned configs | 16.9 | 16.8 |
| this repo's kernel (`apply.py --custom`) | 18.7 | 18.7 |

Spec-off, the paths are within ~15% of each other — decode is
bandwidth-bound and the kernel differences mostly wash out at c1.

## Single node, with the DFlash drafter (2026-07-26)

| configuration | coding c1 | coding c4 agg | coding c8 agg | prose c1 | coding c1 @ 28K-token prompt |
|---|---|---|---|---|---|
| Marlin + drafter | 34.4 | 72.7 | 121.5 | 22.5 | — |
| **custom + drafter** | **37.8** | **82.6** | **123.6** | 22.0 | **33.8** |

The drafter (cross-quant NVFP4 draft, ~2.25 accepted/window) is worth ~2×
end-to-end. Failure modes worth recognizing: ~19 tok/s c1 means the drafter
is off; ~11 tok/s means it accepts 0% (wrong draft/target pairing) and the
wasted draft work is halving throughput.

## Five-node pool, production path (2026-07-26, router → LB → 5 × GB10)

| | c1 | c4 aggregate | c8 aggregate |
|---|---|---|---|
| coding | 40.5 | 100.7 (34.9/stream) | 211.2 (29.6/stream) |
| prose | 23.6 | 73.5 (21.8/stream) | 142.3 (18.5/stream) |

Per-stream decode barely degrades with concurrency: the kernel is *more*
bandwidth-efficient at M=16–64 (226–230 GB/s) than at M=1 (157), so batch
costs far less than the single-stream number suggests.

## Sustained prefill under thermal cap (2026-07-27, 2200 MHz lock)

Stress posture: 4 nodes simultaneously, 9 minutes each, 6 concurrent
streams per node, every request a unique ~4.6K-token prompt (unique prefix,
so zero prefix-cache reuse — every token is real prefill work) plus a
200-token decode.

| | per node | 4-node aggregate |
|---|---|---|
| sustained prefill | **~930 tok/s** | ~3,700 tok/s |
| decode alongside | ~40 tok/s | ~160 tok/s |
| requests / errors | 108–110 / **0** | 436 / **0** |

Thermal context, measured at 10 s resolution during the run: SoC package
zones 92–96 °C (the firmware shutdown trip is a few degrees above), GPU
power 80–89 W per node, software thermal throttle engaging intermittently.
Zero request errors while riding the limiter for 9 minutes, and package
temps recovered 85 °C → 70 °C within 20 s of load end on every node.

### Why the 2200 MHz lock

Prefill is the burn phase on GB10: compute-bound dense GEMM at ~2× the
decode power draw (decode c6 sits near ~50 W; prefill pushes 80–90 W).
Under a 2400 MHz lock in warm ambient, one node's package zones ran away
(96 °C, hard SoC shutdown — silent below the OS); at 2200 MHz + SW
throttle the same load shape rode 92–96 °C stably with zero errors.

**The decode cost of the lower lock is zero, measured** — same node, same
bench, drafter on:

| | 2400 MHz lock | 2200 MHz lock |
|---|---|---|
| coding c1 | 37.8 | 38.3 |
| coding c4 aggregate | 82.6 | 83.2 |
| coding c8 aggregate | 123.6 | 137.4 |
| prose c1 | 22.0 | 23.7 |

Every figure is within (or above) run-to-run variance: decode is
bandwidth-bound, so the SM clock ceiling never binds there — only prefill
pays for clocks, and prefill is precisely the phase that overheats the
package. 2200 is therefore the recommended serving lock for sustained
mixed traffic on this platform unless your ambient is generously cooled.
If you serve heavy-prefill workloads on a GB10, watch the ACPI package
zones (`/sys/class/thermal`), not `nvidia-smi`'s GPU temperature — the
firmware trip follows the zones, which run ~10 °C hotter.
