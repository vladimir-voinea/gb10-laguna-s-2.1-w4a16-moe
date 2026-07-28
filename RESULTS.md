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

## Sustained prefill under thermal cap (2026-07-27, 2200 MHz lock)

Stress posture: 9 minutes, 6 concurrent streams, every request a unique
~4.6K-token prompt (unique prefix, so zero prefix-cache reuse — every token
is real prefill work) plus a 200-token decode.

| | single GB10 |
|---|---|
| sustained prefill | **~930 tok/s** |
| decode alongside | ~40 tok/s |
| requests / errors | 110 / **0** |

Thermal context, measured at 10 s resolution during the run: SoC package
zones 92–96 °C (the firmware shutdown trip is a few degrees above), GPU
power 80–89 W, software thermal throttle engaging intermittently. Zero
request errors while riding the limiter for 9 minutes, and package temps
recovered 85 °C → 70 °C within 20 s of load end.

### Why the 2200 MHz lock

Prefill is the burn phase on GB10: compute-bound dense GEMM at ~2× the
decode power draw (decode c6 sits near ~50 W; prefill pushes 80–90 W).
Under a 2400 MHz lock in warm ambient the package zones can run away
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
package.

**The prefill tax, measured** (same node, engine restarted fresh under each
lock, identical warmup, 75 s of unique ~4.6K-token prompts at c6 with
`max_tokens=16` — near-pure prefill):

| lock | sustained prefill | peak GPU / power during burst |
|---|---|---|
| 2400 MHz | 1,718 tok/s | 85 °C / 94 W (throttle already shaving: clocks dipped to 2288) |
| 2200 MHz | 1,432 tok/s | 83 °C / 80 W, clocks steady |

That is a **~17% prefill cost** — notably more than the 8% the clock ratio
alone predicts, so the graphics lock appears to drag other domains (uncore/
boost) with it on this platform. The trade in practice: prefill is the only
phase that pays, decode is free, and the 2400-lock burst was already
brushing the throttle at 94 W from a cold start — sustained 2400 prefill is
exactly the profile that produced the hard SoC shutdown. Single 75 s run
per point; treat as ±5%. 2200 is therefore the recommended serving lock for sustained
mixed traffic on this platform unless your ambient is generously cooled.
If you serve heavy-prefill workloads on a GB10, watch the ACPI package
zones (`/sys/class/thermal`), not `nvidia-smi`'s GPU temperature — the
firmware trip follows the zones, which run ~10 °C hotter.
