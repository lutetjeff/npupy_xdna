from __future__ import annotations

SUPPORTED_SHAPES: dict[str, list] = {
    "gemm_fusion": [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)],
    "col_indep": [16384, 65536, 262144, 1048576],
    "compute_pool": [32768, 131072, 524288, 2097152],
    "cgra": [256],
}


def assert_shape_supported(template_name: str, shape) -> None:
    if template_name not in SUPPORTED_SHAPES:
        raise ValueError(f"Unknown template: {template_name}")
    if shape not in SUPPORTED_SHAPES[template_name]:
        raise ValueError(
            f"Shape {shape} not in SUPPORTED_SHAPES['{template_name}']. "
            f"Supported: {SUPPORTED_SHAPES[template_name]}"
        )
