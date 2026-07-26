#!/usr/bin/env python3
"""Make vLLM honor `--moe-backend triton` for compressed-tensors WNA16 MoE,
and optionally wire the GB10 custom W4A16 kernel into the serving path.

Stock vLLM 0.25.x hard-prefers the Marlin WNA16 MoE kernel on CUDA whenever
`check_moe_marlin_supports_layer` passes; the Triton int4-w4a16 path is only
reachable when Marlin says no. On GB10 (sm_121a, LPDDR5X) Marlin runs ~4x
below the bandwidth roofline, so the preference is exactly backwards there.

This script patches the installed compressed_tensors_moe.py, in place and
idempotently, so an explicit `--moe-backend triton` opts out of Marlin for
WNA16 MoE layers. No behavior change unless the operator passes the flag.

With ``--custom``, also:

1. Copy ``kernels/w4a16_moe.py`` into site-packages as ``gb10_w4a16_moe.py``
   (site-packages derived from the installed ``vllm`` module path).
2. Patch installed ``compressed_tensors_moe_wna16.py`` so that when
   ``GB10_W4A16_CUSTOM=1``:

   - ``process_weights_after_loading`` repacks post-load w13/w2 qweights+scales
     via ``gb10_w4a16_moe.repack_weights`` and stores them on the layer
     (stock 3-D packed qweights are freed to avoid ~2× expert VRAM; leave
     the env set for the process lifetime).
   - ``apply()`` routes to ``gb10_w4a16_moe.fused_experts_w4a16`` with the
     topk tensors it already has. Fall through to stock when the env is
     unset or repack attrs are absent.

Layout contract (bench ↔ installed post-load)
---------------------------------------------
``create_weights`` stores int32 packs with ``is_transposed=True``:

  w13: (E, K//8, 2N) int32    w2: (E, N//8, K) int32
  scales along the non-transposed group axis.

``process_weights_after_loading`` does ``transpose(1, 2).contiguous().view(uint8)``
which yields the same layout the microbench packs:

  w13: (E, 2N, K//2) uint8    scales (E, 2N, K//group)
  w2:  (E, K,  N//2) uint8    scales (E, K,  N//group)

``repack_weights`` consumes that form directly — no extra transpose.

Run inside the container/venv that serves the model:

    python3 integration/apply.py              # backend patch only
    python3 integration/apply.py --custom     # backend + custom kernel
    python3 integration/apply.py --check      # verify only
    python3 integration/apply.py --revert     # undo all

A ``.orig`` backup is written next to each patched file on first patch.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Backend override (Marlin → honor --moe-backend triton)
# ---------------------------------------------------------------------------

BACKEND_ANCHOR = "or current_platform.is_rocm()"
BACKEND_GUARD = "_gb10_w4a16_moe_backend_override"
BACKEND_INSERT = (
    "                # {guard}: honor an explicit --moe-backend triton for\n"
    "                # WNA16 MoE. On GB10 (sm_121a / LPDDR5X) Marlin runs ~4x\n"
    "                # under the bandwidth roofline; the Triton int4-w4a16 path\n"
    "                # with a device-tuned config is the fast one.\n"
    "                or get_current_vllm_config().kernel_config.moe_backend\n"
    '                == "triton"\n'
).format(guard=BACKEND_GUARD)

# ---------------------------------------------------------------------------
# Custom kernel inject into compressed_tensors_moe_wna16.py
# ---------------------------------------------------------------------------

CUSTOM_GUARD = "_gb10_w4a16_custom_kernel"
KERNEL_MODULE_NAME = "gb10_w4a16_moe.py"

# Inserted after the last scale reassignment in process_weights_after_loading.
PROCESS_ANCHOR = (
    "        layer.w2_weight_scale = torch.nn.Parameter(\n"
    "            layer.w2_weight_scale.transpose(1, 2).contiguous(), "
    "requires_grad=False\n"
    "        )\n"
)

PROCESS_INSERT = """\
        # {guard}: opt-in GB10 streaming-layout repack (env GB10_W4A16_CUSTOM=1).
        # Layout after the transpose/view above matches the microbench packs —
        # (E, N_out, K_in//2) uint8 + (E, N_out, K_in//group) scales. create_weights
        # stored is_transposed=True int32; that transpose is the only fix needed.
        # repack_weights takes this post-load layout with no further transpose.
        import os as _gb10_os
        if _gb10_os.environ.get("GB10_W4A16_CUSTOM") == "1":
            import gb10_w4a16_moe as _gb10_k
            _w1q, _w1s = _gb10_k.repack_weights(
                layer.w13_weight_packed.data, layer.w13_weight_scale.data
            )
            _w2q, _w2s = _gb10_k.repack_weights(
                layer.w2_weight_packed.data, layer.w2_weight_scale.data
            )
            # Bind fused entry on the layer so apply() has zero import/lookup cost.
            layer._gb10_w1_q = _w1q
            layer._gb10_w1_scale = _w1s
            layer._gb10_w2_q = _w2q
            layer._gb10_w2_scale = _w2s
            layer._gb10_group_size = int(self.group_size)
            layer._gb10_fused = _gb10_k.fused_experts_w4a16
            # Free stock 3-D packed qweights (repacked is the serving path).
            # Keeping both would ~2× MoE expert VRAM. Scales stay for quant_config
            # readers. Stock fallthrough needs repack absent (env unset at load);
            # leave GB10_W4A16_CUSTOM=1 for the process lifetime when using custom.
            _dev = _w1q.device
            layer.w13_weight_packed = torch.nn.Parameter(
                torch.empty(0, dtype=torch.uint8, device=_dev), requires_grad=False
            )
            layer.w2_weight_packed = torch.nn.Parameter(
                torch.empty(0, dtype=torch.uint8, device=_dev), requires_grad=False
            )
            # Free the STOCK SCALES too. repack_weights emits its own tiled copy,
            # so keeping both duplicates ~150 MiB per layer -- about 7 GiB across
            # 47 layers, taken straight out of the KV pool. Measured: model load
            # 76.32 GiB against a 67 GB checkpoint, KV 479k tokens (1.83x at 262K)
            # versus NVFP4's 794k (3.03x). Safe because apply() routes entirely to
            # the custom kernel when the env is set; nothing reads these again.
            layer.w13_weight_scale = torch.nn.Parameter(
                torch.empty(0, dtype=_w1s.dtype, device=_dev), requires_grad=False
            )
            layer.w2_weight_scale = torch.nn.Parameter(
                torch.empty(0, dtype=_w2s.dtype, device=_dev), requires_grad=False
            )
            # Hand the blocks back to the driver, not just to torch's caching
            # allocator: vLLM sizes the KV cache from free DEVICE memory.
            import gc as _gb10_gc
            _gb10_gc.collect()
            torch.cuda.empty_cache()
""".format(guard=CUSTOM_GUARD)

# Inserted at the top of apply() body, before the stock fused_experts import.
APPLY_ANCHOR = (
    "    ) -> torch.Tensor:\n"
    "        from vllm.model_executor.layers.fused_moe import fused_experts\n"
)

APPLY_INSERT = (
    "    ) -> torch.Tensor:\n"
    "        # {guard}: hot path — precomputed repacked weights, no alloc here.\n"
    "        import os as _gb10_os\n"
    "        _gb10_fn = getattr(layer, \"_gb10_fused\", None)\n"
    "        if (\n"
    "            _gb10_fn is not None\n"
    '            and _gb10_os.environ.get("GB10_W4A16_CUSTOM") == "1"\n'
    "        ):\n"
    "            return _gb10_fn(\n"
    "                x,\n"
    "                layer._gb10_w1_q,\n"
    "                layer._gb10_w2_q,\n"
    "                layer._gb10_w1_scale,\n"
    "                layer._gb10_w2_scale,\n"
    "                topk_weights,\n"
    "                topk_ids,\n"
    "                group_size=layer._gb10_group_size,\n"
    "            )\n"
    "        from vllm.model_executor.layers.fused_moe import fused_experts\n"
).format(guard=CUSTOM_GUARD)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _kernel_src() -> Path:
    return _repo_root() / "kernels" / "w4a16_moe.py"


def find_site_packages() -> Path:
    """site-packages that owns the installed ``vllm`` package."""
    import vllm

    # vllm/__init__.py → parent is the vllm package dir → parent is site-packages
    return Path(vllm.__file__).resolve().parent.parent


def find_backend_target() -> Path:
    import vllm.model_executor.layers.quantization.compressed_tensors as ct

    root = Path(ct.__file__).parent
    f = root / "compressed_tensors_moe" / "compressed_tensors_moe.py"
    if not f.exists():
        sys.exit(f"not found: {f}")
    return f


def find_wna16_target() -> Path:
    import vllm.model_executor.layers.quantization.compressed_tensors as ct

    root = Path(ct.__file__).parent
    candidates = [
        root / "compressed_tensors_moe" / "compressed_tensors_moe_wna16.py",
        root / "compressed_tensors_moe_wna16.py",
    ]
    for f in candidates:
        if f.exists():
            return f
    sys.exit(
        "compressed_tensors_moe_wna16.py not found under "
        f"{root}; tried: {', '.join(str(c) for c in candidates)}"
    )


def _backup_path(f: Path) -> Path:
    return f.with_suffix(".py.orig")


def _patch_backend(check_only: bool) -> tuple[bool, str]:
    f = find_backend_target()
    text = f.read_text()
    patched = BACKEND_GUARD in text
    if check_only:
        return patched, f"{f}: {'patched' if patched else 'NOT patched'}"
    if patched:
        return True, f"backend already patched: {f}"
    if BACKEND_ANCHOR not in text:
        sys.exit(
            f"backend anchor not found — vLLM version drift; patch by hand near "
            f"the WNA16 Marlin-preference check in {f}"
        )
    if "get_current_vllm_config" not in text:
        sys.exit("get_current_vllm_config not imported in backend target module")
    backup = _backup_path(f)
    if not backup.exists():
        shutil.copy2(f, backup)
    new = text.replace(BACKEND_ANCHOR + "\n", BACKEND_ANCHOR + "\n" + BACKEND_INSERT, 1)
    f.write_text(new)
    return True, f"patched backend {f} (backup: {backup})"


def _patch_wna16(check_only: bool) -> tuple[bool, str]:
    f = find_wna16_target()
    text = f.read_text()
    patched = CUSTOM_GUARD in text
    if check_only:
        return patched, f"{f}: {'custom-patched' if patched else 'NOT custom-patched'}"
    if patched:
        return True, f"wna16 already custom-patched: {f}"
    if PROCESS_ANCHOR not in text:
        sys.exit(
            f"process_weights anchor not found in {f} — vLLM version drift; "
            "compare with reference/compressed_tensors_moe_wna16.py"
        )
    if APPLY_ANCHOR not in text:
        sys.exit(
            f"apply() anchor not found in {f} — vLLM version drift; "
            "compare with reference/compressed_tensors_moe_wna16.py"
        )
    backup = _backup_path(f)
    if not backup.exists():
        shutil.copy2(f, backup)
    new = text.replace(PROCESS_ANCHOR, PROCESS_ANCHOR + PROCESS_INSERT, 1)
    new = new.replace(APPLY_ANCHOR, APPLY_INSERT, 1)
    if CUSTOM_GUARD not in new:
        sys.exit(f"custom patch failed to apply cleanly to {f}")
    f.write_text(new)
    return True, f"patched wna16 {f} (backup: {backup})"


def _install_kernel(check_only: bool) -> tuple[bool, str]:
    src = _kernel_src()
    if not src.exists():
        sys.exit(f"kernel source missing: {src}")
    dest = find_site_packages() / KERNEL_MODULE_NAME
    if check_only:
        ok = dest.exists()
        return ok, f"{dest}: {'installed' if ok else 'NOT installed'}"
    shutil.copy2(src, dest)
    return True, f"installed kernel {src} -> {dest}"


def _revert_file(f: Path) -> str:
    backup = _backup_path(f)
    if backup.exists():
        shutil.copy2(backup, f)
        return f"reverted {f} from {backup}"
    if f.exists():
        # No backup but file present — leave it; report.
        return f"no .orig backup for {f}; left in place"
    return f"missing {f}"


def revert_all() -> int:
    messages: list[str] = []
    # Backend
    try:
        messages.append(_revert_file(find_backend_target()))
    except SystemExit as e:
        messages.append(f"backend: {e}")
    # WNA16
    try:
        messages.append(_revert_file(find_wna16_target()))
    except SystemExit as e:
        messages.append(f"wna16: {e}")
    # Kernel module
    dest = find_site_packages() / KERNEL_MODULE_NAME
    if dest.exists():
        dest.unlink()
        messages.append(f"removed {dest}")
    else:
        messages.append(f"kernel module absent: {dest}")
    for m in messages:
        print(m)
    return 0


def check_all(*, want_custom: bool) -> int:
    ok_b, msg_b = _patch_backend(check_only=True)
    print(msg_b)
    status = 0 if ok_b else 1
    if want_custom:
        ok_k, msg_k = _install_kernel(check_only=True)
        print(msg_k)
        ok_w, msg_w = _patch_wna16(check_only=True)
        print(msg_w)
        if not (ok_k and ok_w):
            status = 1
    else:
        # Still report custom state if present, but don't fail on absence.
        try:
            _, msg_k = _install_kernel(check_only=True)
            print(msg_k)
            _, msg_w = _patch_wna16(check_only=True)
            print(msg_w)
        except SystemExit as e:
            print(f"custom targets: {e}")
    return status


def apply_all(*, custom: bool) -> int:
    _, msg_b = _patch_backend(check_only=False)
    print(msg_b)
    if custom:
        _, msg_k = _install_kernel(check_only=False)
        print(msg_k)
        _, msg_w = _patch_wna16(check_only=False)
        print(msg_w)
        print(
            "custom kernel ready — serve with GB10_W4A16_CUSTOM=1 "
            "(and --moe-backend triton)."
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--custom",
        action="store_true",
        help="also install gb10_w4a16_moe.py and patch WNA16 apply/repack path",
    )
    ap.add_argument("--check", action="store_true", help="verify patch state only")
    ap.add_argument("--revert", action="store_true", help="restore .orig backups")
    args = ap.parse_args()

    if args.revert:
        return revert_all()
    if args.check:
        return check_all(want_custom=args.custom)
    return apply_all(custom=args.custom)


if __name__ == "__main__":
    raise SystemExit(main())
