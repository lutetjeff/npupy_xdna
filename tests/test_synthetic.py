import numpy as np
import pytest

from npupy_xdna.bench.synthetic.horner_polynomial import horner_eval, make_polynomial_coeffs


class TestHornerEval:
    def test_quadratic_exact(self):
        x = np.array([2], dtype=np.int16)
        coeffs = [1, 2, 3]
        result = horner_eval(x, coeffs)
        assert result.dtype == np.int16
        assert result[0] == 17  # 1 + 2*2 + 3*4

    def test_constant_polynomial(self):
        x = np.array([0], dtype=np.int16)
        coeffs = [5]
        result = horner_eval(x, coeffs)
        assert result.dtype == np.int16
        assert result[0] == 5

    def test_degree_4(self):
        rng = np.random.default_rng(123)
        x = rng.integers(-100, 101, size=64, dtype=np.int16)
        coeffs = make_polynomial_coeffs(4, seed=1)
        result = horner_eval(x, coeffs)
        assert result.dtype == np.int16
        assert len(result) == 64

    def test_degree_8(self):
        rng = np.random.default_rng(456)
        x = rng.integers(-50, 51, size=128, dtype=np.int16)
        coeffs = make_polynomial_coeffs(8, seed=2)
        result = horner_eval(x, coeffs)
        assert result.dtype == np.int16
        assert len(result) == 128
        assert len(coeffs) == 9

    def test_degree_32_no_overflow(self):
        rng = np.random.default_rng(789)
        x = rng.integers(-10, 11, size=1000, dtype=np.int16)
        coeffs = make_polynomial_coeffs(32, seed=3)
        result = horner_eval(x, coeffs)
        assert result.dtype == np.int16
        assert len(result) == 1000
        assert np.all(result >= -32768)
        assert np.all(result <= 32767)

    def test_negative_input(self):
        x = np.array([-3], dtype=np.int16)
        coeffs = [1, -2, 1]  # 1 - 2x + x^2
        result = horner_eval(x, coeffs)
        assert result[0] == 16  # 1 - 2*(-3) + (-3)^2 = 1 + 6 + 9 = 16

    def test_clipping(self):
        x = np.array([30000], dtype=np.int16)
        coeffs = [0, 0, 2]  # 2*x^2 will overflow int32 but clip to int16
        result = horner_eval(x, coeffs)
        assert result.dtype == np.int16
        assert np.all(result >= -32768)
        assert np.all(result <= 32767)


class TestMakePolynomialCoeffs:
    def test_length(self):
        coeffs = make_polynomial_coeffs(8)
        assert len(coeffs) == 9

    def test_reproducibility(self):
        c1 = make_polynomial_coeffs(10, seed=42)
        c2 = make_polynomial_coeffs(10, seed=42)
        assert c1 == c2

    def test_range(self):
        coeffs = make_polynomial_coeffs(100, seed=99)
        assert all(-3 <= c <= 3 for c in coeffs)

    def test_degree_32(self):
        coeffs = make_polynomial_coeffs(32)
        assert len(coeffs) == 33
