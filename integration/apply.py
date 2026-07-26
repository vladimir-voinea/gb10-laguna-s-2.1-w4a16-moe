#!/usr/bin/env python3
"""Make vLLM honor `--moe-backend triton` for compressed-tensors WNA16 MoE.

Stock vLLM 0.25.x hard-prefers the Marlin WNA16 MoE kernel on CUDA whenever
`check_moe_marlin_supports_layer` passes; the Triton int4-w4a16 path is only
reachable when Marlin says no. On GB10 (sm_121a, LPDDR5X) Marlin runs ~4x
below the bandwidth roofline, so the preference is exactly backwards there.

This script patches the installed compressed_tensors_moe.py, in place and
idempotently, so an explicit `--moe-backend triton` opts out of Marlin for
WNA16 MoE layers. No behavior change unless the operator passes the flag.

Run inside the container/venv that serves the model:

    python3 integration/apply.py            # patch
    python3 integration/apply.py --check    # verify only
    python3 integration/apply.py --revert   # undo

A `.orig` backup is written next to the file on first patch.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ANCHOR = "or current_platform.is_rocm()"
GUARD = "_gb10_w4a16_moe_backend_override"
INSERT = (
    "                # {guard}: honor an explicit --moe-backend triton for\n"
    "                # WNA16 MoE. On GB10 (sm_121a / LPDDR5X) Marlin runs ~4x\n"
    "                # under the bandwidth roofline; the Triton int4-w4a16 path\n"
    "                # with a device-tuned config is the fast one.\n"
    "                or get_current_vllm_config().kernel_config.moe_backend\n"
    '                == "triton"\n'
).format(guard=GUARD)


def find_target() -> Path:
    import vllm.model_executor.layers.quantization.compressed_tensors as ct

    root = Path(ct.__file__).parent
    f = root / "compressed_tensors_moe" / "compressed_tensors_moe.py"
    if not f.exists():
        sys.exit(f"not found: {f}")
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    f = find_target()
    text = f.read_text()
    patched = GUARD in text

    if args.check:
        print(f"{f}: {'patched' if patched else 'NOT patched'}")
        return 0 if patched else 1

    backup = f.with_suffix(".py.orig")
    if args.revert:
        if backup.exists():
            shutil.copy2(backup, f)
            print(f"reverted from {backup}")
            return 0
        sys.exit("no .orig backup to revert from")

    if patched:
        print("already patched")
        return 0
    if ANCHOR not in text:
        sys.exit(
            f"anchor line not found — vLLM version drift; patch by hand near "
            f"the WNA16 Marlin-preference check in {f}"
        )
    if "get_current_vllm_config" not in text.split("def ")[0]:
        # the file already imports it for the ROCm branch in 0.25.x; if a
        # future version dropped it, fail loudly rather than emit a NameError.
        if "get_current_vllm_config" not in text:
            sys.exit("get_current_vllm_config not imported in target module")

    shutil.copy2(f, backup)
    new = text.replace(
        ANCHOR + "\n",
        ANCHOR + "\n" + INSERT,
        1,
    )
    f.write_text(new)
    print(f"patched {f} (backup: {backup})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
