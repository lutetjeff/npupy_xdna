from __future__ import annotations

import numpy as np
import pytest

from npupy_xdna.regions.region import ArraySpec, Region
from npupy_xdna.runtime.cpu_runner import CpuRunner


def test_cpu_runner_matmul_bitexact():
    rng = np.random.default_rng(42)
    A = rng.integers(0, 10, size=(64, 64), dtype=np.int16)
    B = rng.integers(0, 10, size=(64, 64), dtype=np.int16)
    region = Region(
        op="matmul",
        inputs=[
            ArraySpec(shape=(64, 64), dtype="int16"),
            ArraySpec(shape=(64, 64), dtype="int16"),
        ],
        output=ArraySpec(shape=(64, 64), dtype="int16"),
    )
    runner = CpuRunner()
    result = runner.run(region, [A, B])
    expected = np.matmul(A, B)
    assert np.array_equal(result.output, expected)
    assert result.status == "ok"
    assert result.device == "cpu"
    assert result.latency_us >= 0


def test_cpu_runner_elementwise_unary_relu():
    A = np.array([[-1, 2], [3, -4]], dtype=np.int16)
    region = Region(
        op="elementwise_unary",
        inputs=[ArraySpec(shape=(2, 2), dtype="int16")],
        output=ArraySpec(shape=(2, 2), dtype="int16"),
    )
    runner = CpuRunner()
    result = runner.run(region, [A])
    expected = np.maximum(0, A)
    assert np.array_equal(result.output, expected)


def test_cpu_runner_elementwise_binary_add():
    A = np.array([[1, 2], [3, 4]], dtype=np.int16)
    B = np.array([[5, 6], [7, 8]], dtype=np.int16)
    region = Region(
        op="elementwise_binary",
        inputs=[
            ArraySpec(shape=(2, 2), dtype="int16"),
            ArraySpec(shape=(2, 2), dtype="int16"),
        ],
        output=ArraySpec(shape=(2, 2), dtype="int16"),
    )
    runner = CpuRunner()
    result = runner.run(region, [A, B])
    expected = A + B
    assert np.array_equal(result.output, expected)


def test_cpu_runner_matmul_fused_relu():
    rng = np.random.default_rng(42)
    A = rng.integers(0, 5, size=(8, 8), dtype=np.int16)
    B = rng.integers(-5, 5, size=(8, 8), dtype=np.int16)
    region = Region(
        op="matmul_fused",
        inputs=[
            ArraySpec(shape=(8, 8), dtype="int16"),
            ArraySpec(shape=(8, 8), dtype="int16"),
        ],
        output=ArraySpec(shape=(8, 8), dtype="int16"),
    )
    runner = CpuRunner()
    result = runner.run(region, [A, B])
    expected = np.maximum(0, np.matmul(A, B))
    assert np.array_equal(result.output, expected)


def test_cpu_runner_chained_elementwise():
    A = np.array([[1, -2], [3, -4]], dtype=np.int16)
    region = Region(
        op="chained_elementwise",
        inputs=[ArraySpec(shape=(2, 2), dtype="int16")],
        output=ArraySpec(shape=(2, 2), dtype="int16"),
    )
    runner = CpuRunner()
    result = runner.run(region, [A])
    expected = np.maximum(0, A + 1)
    assert np.array_equal(result.output, expected)


def test_cpu_runner_unsupported_op():
    A = np.array([[1, 2]], dtype=np.int16)
    B = np.array([[3], [4]], dtype=np.int16)
    runner = CpuRunner()
    with pytest.raises(ValueError, match="Unsupported op"):
        runner.run(
            Region(
                op="conv2d",
                inputs=[
                    ArraySpec(shape=(1, 2), dtype="int16"),
                    ArraySpec(shape=(2, 1), dtype="int16"),
                ],
                output=ArraySpec(shape=(1, 1), dtype="int16"),
            ),
            [A, B],
        )


def test_cpu_runner_wrong_input_count():
    A = np.array([[1, 2]], dtype=np.int16)
    runner = CpuRunner()
    region = Region(
        op="matmul",
        inputs=[
            ArraySpec(shape=(1, 2), dtype="int16"),
            ArraySpec(shape=(2, 1), dtype="int16"),
        ],
        output=ArraySpec(shape=(1, 1), dtype="int16"),
    )
    with pytest.raises(ValueError, match="matmul requires exactly 2 inputs"):
        runner.run(region, [A])


def test_cpu_runner_wrong_input_count():
    A = np.array([[1, 2]], dtype=np.int16)
    runner = CpuRunner()
    region = Region(
        op="matmul",
        inputs=[
            ArraySpec(shape=(1, 2), dtype="int16"),
            ArraySpec(shape=(2, 1), dtype="int16"),
        ],
        output=ArraySpec(shape=(1, 1), dtype="int16"),
    )
    with pytest.raises(ValueError, match="matmul requires exactly 2 inputs"):
        runner.run(region, [A])
