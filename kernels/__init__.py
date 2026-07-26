# Custom W4A16 MoE kernels for GB10.
# Import from kernels.w4a16_moe directly to avoid pulling triton at package import.
__all__ = ["fused_experts_w4a16", "repack_weights"]


def __getattr__(name: str):
    if name in __all__:
        from kernels import w4a16_moe

        return getattr(w4a16_moe, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
