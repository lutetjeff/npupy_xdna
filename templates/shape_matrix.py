from __future__ import annotations

SUPPORTED_SHAPES: dict[str, list] = {
    "gemm_fusion": [(128, 128, 128), (256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)],
    "col_indep": [16384, 65536, 262144, 1048576, 2097152, 4194304],
    "compute_pool": [32768, 131072, 524288, 2097152],
    "cgra": [256],
    # sliding_window: (H, W) 2D grids for 5-point stencil; H must be divisible by 8 (num columns)
    "sliding_window": [(64, 64), (128, 128), (256, 256)],
    "tanh": [65536, 262144, 1048576, 4194304],
    "hash": [65536, 262144, 1048576],
}


def assert_shape_supported(template_name: str, shape) -> None:
    if template_name not in SUPPORTED_SHAPES:
        raise ValueError(f"Unknown template: {template_name}")
    if shape not in SUPPORTED_SHAPES[template_name]:
        raise ValueError(
            f"Shape {shape} not in SUPPORTED_SHAPES['{template_name}']. "
            f"Supported: {SUPPORTED_SHAPES[template_name]}"
        )
