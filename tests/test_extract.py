import operator

import numpy as np
import pytest

from npupy_xdna.dispatch.extract import numpy_op_to_region
from npupy_xdna.regions.region import ArraySpec, Region


def i16(*shape):
    return np.zeros(shape, dtype=np.int16)


class TestMatmul:
    def test_np_matmul_produces_matmul_region(self):
        A = i16(2, 3)
        B = i16(3, 4)
        r = numpy_op_to_region(np.matmul, (A, B), {})
        assert r is not None
        assert r.op == "matmul"
        assert r.inputs[0] == ArraySpec(shape=(2, 3), dtype="int16")
        assert r.inputs[1] == ArraySpec(shape=(3, 4), dtype="int16")
        assert r.output == ArraySpec(shape=(2, 4), dtype="int16")

    def test_operator_matmul_produces_matmul_region(self):
        A = i16(3, 3)
        B = i16(3, 3)
        r = numpy_op_to_region(operator.matmul, (A, B), {})
        assert r is not None
        assert r.op == "matmul"

    def test_inner_dim_mismatch_returns_none(self):
        A = i16(2, 3)
        B = i16(4, 2)
        assert numpy_op_to_region(np.matmul, (A, B), {}) is None

    def test_non_int16_returns_none(self):
        A = np.zeros((2, 2), dtype=np.float32)
        B = np.zeros((2, 2), dtype=np.float32)
        assert numpy_op_to_region(np.matmul, (A, B), {}) is None

    def test_1d_returns_none(self):
        A = i16(4)
        B = i16(4)
        assert numpy_op_to_region(np.matmul, (A, B), {}) is None

    def test_output_shape_is_correct(self):
        A = i16(5, 7)
        B = i16(7, 3)
        r = numpy_op_to_region(np.matmul, (A, B), {})
        assert r.output.shape == (5, 3)


class TestAdd:
    def test_np_add_produces_elementwise_binary(self):
        A = i16(4, 4)
        B = i16(4, 4)
        r = numpy_op_to_region(np.add, (A, B), {})
        assert r is not None
        assert r.op == "elementwise_binary"

    def test_add_1d_arrays(self):
        A = i16(8)
        B = i16(8)
        r = numpy_op_to_region(np.add, (A, B), {})
        assert r is not None
        assert r.op == "elementwise_binary"
        assert r.inputs[0].shape == (8,)

    def test_add_non_int16_returns_none(self):
        A = np.zeros(4, dtype=np.float64)
        B = np.zeros(4, dtype=np.float64)
        assert numpy_op_to_region(np.add, (A, B), {}) is None


class TestMultiply:
    def test_np_multiply_produces_elementwise_binary(self):
        A = i16(3, 3)
        B = i16(3, 3)
        r = numpy_op_to_region(np.multiply, (A, B), {})
        assert r is not None
        assert r.op == "elementwise_binary"

    def test_multiply_non_int16_returns_none(self):
        A = np.zeros((3, 3), dtype=np.int32)
        B = np.zeros((3, 3), dtype=np.int32)
        assert numpy_op_to_region(np.multiply, (A, B), {}) is None


class TestMaximum:
    def test_relu_pattern_zero_first(self):
        x = i16(4, 4)
        r = numpy_op_to_region(np.maximum, (0, x), {})
        assert r is not None
        assert r.op == "elementwise_unary"
        assert r.inputs[0].shape == (4, 4)
        assert r.output.shape == (4, 4)

    def test_relu_pattern_zero_second(self):
        x = i16(4, 4)
        r = numpy_op_to_region(np.maximum, (x, 0), {})
        assert r is not None
        assert r.op == "elementwise_unary"

    def test_relu_pattern_float_zero(self):
        x = i16(4)
        r = numpy_op_to_region(np.maximum, (0.0, x), {})
        assert r is not None
        assert r.op == "elementwise_unary"

    def test_general_binary_maximum(self):
        A = i16(4, 4)
        B = i16(4, 4)
        r = numpy_op_to_region(np.maximum, (A, B), {})
        assert r is not None
        assert r.op == "elementwise_binary"

    def test_maximum_non_int16_returns_none(self):
        x = np.zeros(4, dtype=np.float32)
        assert numpy_op_to_region(np.maximum, (0, x), {}) is None


class TestUnsupportedOps:
    def test_linalg_eig_returns_none(self):
        A = i16(2, 2)
        assert numpy_op_to_region(np.linalg.eig, (A,), {}) is None

    def test_sum_returns_none(self):
        A = i16(4)
        assert numpy_op_to_region(np.sum, (A,), {}) is None

    def test_sin_returns_none(self):
        A = i16(4)
        assert numpy_op_to_region(np.sin, (A,), {}) is None

    def test_subtract_returns_none(self):
        A = i16(4)
        B = i16(4)
        assert numpy_op_to_region(np.subtract, (A, B), {}) is None
