import torch
import copy
import traceback
from dataclasses import dataclass
from typing import List
from typing import Callable, List

# =========================================================
# Fake batch for AlphaFold-like shapes
# =========================================================
torch.manual_seed(11)


def make_fake_msa_batch(
    B=2,
    N_msa=128,
    L=250,
    c_m=256,
    device="cpu",
    dtype=torch.float32,
):
    m = torch.randn(B, N_msa, L, c_m, device=device, dtype=dtype)
    msa_mask = torch.ones(B, N_msa, L, device=device, dtype=dtype)

    # padding artificial al final
    for b in range(B):
        cut = torch.randint(low=int(0.7 * L), high=L + 1, size=(1,)).item()
        msa_mask[b, :, cut:] = 0.0

    return {
        "m": m,
        "msa_mask": msa_mask,
    }


# =========================================================
# Helpers
# =========================================================
def assert_shape(x, expected_shape, name="tensor"):
    assert torch.is_tensor(x), f"{name} must be a tensor"
    assert tuple(x.shape) == tuple(expected_shape), (
        f"{name} has shape {tuple(x.shape)} but expected {tuple(expected_shape)}"
    )


def assert_finite_tensor(x, name="tensor"):
    assert torch.is_tensor(x), f"{name} must be a tensor"
    assert torch.isfinite(x).all(), f"{name} contains NaN or Inf"


def assert_close(x, y, atol=1e-6, rtol=1e-5, name="tensor"):
    assert torch.allclose(x, y, atol=atol, rtol=rtol), f"{name} not close"


def assert_not_close(x, y, atol=1e-6, rtol=1e-5, name="tensor"):
    assert not torch.allclose(x, y, atol=atol, rtol=rtol), f"{name} unexpectedly equal"


def assert_scalar_finite(x, name="scalar"):
    assert torch.is_tensor(x), f"{name} must be a tensor"
    assert x.ndim == 0, f"{name} must be scalar, got shape {tuple(x.shape)}"
    assert torch.isfinite(x), f"{name} is not finite"


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str = ""


def run_test_silent(name: str, fn: Callable[[], None]) -> TestResult:
    try:
        fn()
        return TestResult(name=name, passed=True, message="")
    except Exception as e:
        tb = traceback.format_exc()
        return TestResult(
            name=name,
            passed=False,
            message=f"{type(e).__name__}: {e}\n{tb}"
        )


def finalize_test_results(results: List[TestResult], suite_name="test_suite"):
    failures = [r for r in results if not r.passed]

    if not failures:
        return

    print(f"[{suite_name}] {len(failures)} test(s) failed:\n")
    for r in failures:
        print(f"[FAIL] {r.name}")
        print(r.message)
        print("-" * 100)

    raise AssertionError(f"{suite_name} failed with {len(failures)} failing test(s)")