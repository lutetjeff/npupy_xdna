from __future__ import annotations

import pytest

from npupy_xdna.regions.region import ArraySpec, Region, SUPPORTED_OPS


def test_matmul_int16_accepted():
    region = Region(
        op="matmul",
        inputs=[
            ArraySpec(shape=(4, 5), dtype="int16"),
            ArraySpec(shape=(5, 6), dtype="int16"),
        ],
        output=ArraySpec(shape=(4, 6), dtype="int16"),
    )
    assert region.op == "matmul"
    assert region.inputs[0].shape == (4, 5)


def test_matmul_bf16_rejected():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        Region(
            op="matmul",
            inputs=[
                ArraySpec(shape=(4, 5), dtype="bf16"),
                ArraySpec(shape=(5, 6), dtype="bf16"),
            ],
            output=ArraySpec(shape=(4, 6), dtype="bf16"),
        )


def test_k_dim_mismatch():
    with pytest.raises(ValueError, match="Inner dimensions must match"):
        Region(
            op="matmul",
            inputs=[
                ArraySpec(shape=(4, 5), dtype="int16"),
                ArraySpec(shape=(6, 7), dtype="int16"),
            ],
            output=ArraySpec(shape=(4, 7), dtype="int16"),
        )


def test_zero_dim_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        ArraySpec(shape=(0, 5), dtype="int16")


def test_json_roundtrip():
    region = Region(
        op="matmul_fused",
        inputs=[
            ArraySpec(shape=(8, 8), dtype="int16"),
            ArraySpec(shape=(8, 8), dtype="int16"),
        ],
        output=ArraySpec(shape=(8, 8), dtype="int16"),
        metadata={"tag": "test"},
    )
    s = region.to_json()
    restored = Region.from_json(s)
    assert restored.op == region.op
    assert restored.inputs[0].shape == (8, 8)
    assert restored.inputs[0].dtype == "int16"
    assert restored.output.shape == (8, 8)
    assert restored.output.dtype == "int16"
    assert restored.metadata == {"tag": "test"}


def test_all_supported_ops():
    for op in SUPPORTED_OPS:
        if op in ("matmul", "matmul_fused"):
            region = Region(
                op=op,
                inputs=[
                    ArraySpec(shape=(2, 2), dtype="int16"),
                    ArraySpec(shape=(2, 2), dtype="int16"),
                ],
                output=ArraySpec(shape=(2, 2), dtype="int16"),
            )
        elif op in ("elementwise_unary", "chained_elementwise"):
            region = Region(
                op=op,
                inputs=[ArraySpec(shape=(2, 2), dtype="int16")],
                output=ArraySpec(shape=(2, 2), dtype="int16"),
            )
        elif op == "elementwise_binary":
            region = Region(
                op=op,
                inputs=[
                    ArraySpec(shape=(2, 2), dtype="int16"),
                    ArraySpec(shape=(2, 2), dtype="int16"),
                ],
                output=ArraySpec(shape=(2, 2), dtype="int16"),
            )
        else:
            continue
        assert region.op == op


def test_unsupported_op_rejected():
    with pytest.raises(ValueError, match="Unsupported op"):
        Region(
            op="conv2d",
            inputs=[ArraySpec(shape=(2, 2), dtype="int16")],
            output=ArraySpec(shape=(2, 2), dtype="int16"),
        )


def test_output_shape_mismatch():
    with pytest.raises(ValueError, match="Output shape mismatch"):
        Region(
            op="matmul",
            inputs=[
                ArraySpec(shape=(4, 5), dtype="int16"),
                ArraySpec(shape=(5, 6), dtype="int16"),
            ],
            output=ArraySpec(shape=(4, 7), dtype="int16"),
        )
