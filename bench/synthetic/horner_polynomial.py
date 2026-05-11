import numpy as np


def horner_eval(x: np.ndarray, coeffs: list[int], dtype: type = np.int16) -> np.ndarray:
    """Evaluate polynomial via Horner's method: coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...

    Uses int32 accumulator to avoid overflow, clips to int16 at end.

    Parameters
    ----------
    x : np.ndarray
        Input values (typically int16).
    coeffs : list[int]
        Polynomial coefficients [c0, c1, c2, ...] for c0 + c1*x + c2*x^2 + ...
    dtype : numpy dtype, optional
        Output dtype (default np.int16).

    Returns
    -------
    np.ndarray
        Evaluated polynomial values, clipped to int16 range.
    """
    acc = np.zeros_like(x, dtype=np.int32)
    for c in reversed(coeffs):
        acc = acc * x.astype(np.int32) + c
    return np.clip(acc, -32768, 32767).astype(dtype)


def make_polynomial_coeffs(degree: int, seed: int = 42) -> list[int]:
    """Generate small-magnitude int16 coefficients for a polynomial of given degree.

    Parameters
    ----------
    degree : int
        Polynomial degree (returns degree + 1 coefficients).
    seed : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    list[int]
        Coefficients in range [-3, 3].
    """
    rng = np.random.default_rng(seed)
    return [int(v) for v in rng.integers(-3, 4, size=degree + 1)]
