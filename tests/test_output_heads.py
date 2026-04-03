import torch
import torch.nn as nn

from model.alphafold2_heads import *
from model.torsion_head import *

torch.manual_seed(123)

def assert_close(x, y, atol=1e-5, rtol=1e-5, msg=""):
    if not torch.allclose(x, y, atol=atol, rtol=rtol):
        max_err = (x - y).abs().max().item()
        raise AssertionError(f"{msg} | max_err={max_err}")

def test_single_projection():
    B, N, L, c_m, c_s = 2, 16, 37, 256, 256
    m = torch.randn(B, N, L, c_m)

    mod = SingleProjection(c_m=c_m, c_s=c_s)
    mod.eval()

    s = mod(m)

    assert s.shape == (B, L, c_s), f"Bad shape: {s.shape}"
    assert torch.isfinite(s).all(), "SingleProjection has non-finite values"

    # check that it is really using first row only
    m2 = m.clone()
    m2[:, 1:] = torch.randn_like(m2[:, 1:]) * 1000.0
    s2 = mod(m2)

    assert_close(s, s2, atol=1e-5, msg="SingleProjection should depend only on first MSA row")
    print("OK: SingleProjection")


def test_plddt_head():
    B, L, c_s = 2, 51, 256
    num_bins = 50
    s = torch.randn(B, L, c_s)

    mod = PlddtHead(c_s=c_s, hidden=256, num_bins=num_bins)
    mod.eval()

    logits, plddt = mod(s)

    assert logits.shape == (B, L, num_bins), f"Bad logits shape: {logits.shape}"
    assert plddt.shape == (B, L), f"Bad plddt shape: {plddt.shape}"

    assert torch.isfinite(logits).all(), "pLDDT logits contain non-finite values"
    assert torch.isfinite(plddt).all(), "pLDDT contains non-finite values"

    # Since pLDDT is expectation over bins in [0,100], it must stay in [0,100]
    assert (plddt >= 0.0).all(), "pLDDT has values < 0"
    assert (plddt <= 100.0).all(), "pLDDT has values > 100"

    # softmax sanity
    probs = torch.softmax(logits, dim=-1)
    sums = probs.sum(dim=-1)
    assert_close(sums, torch.ones_like(sums), atol=1e-5, msg="Softmax probs do not sum to 1")

    print("OK: PlddtHead")


def test_distogram_head():
    B, L, c_z, num_bins = 2, 41, 128, 64
    z = torch.randn(B, L, L, c_z)

    mod = DistogramHead(c_z=c_z, num_bins=num_bins)
    mod.eval()

    logits = mod(z)

    assert logits.shape == (B, L, L, num_bins), f"Bad shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "Distogram logits contain non-finite values"

    # symmetry check
    assert_close(
        logits,
        logits.transpose(1, 2),
        atol=1e-5,
        msg="Distogram logits should be symmetric in residue pair indices"
    )

    print("OK: DistogramHead")


def test_torsion_resblock():
    B, L, dim = 2, 29, 256
    x = torch.randn(B, L, dim)

    mod = TorsionResBlock(dim=dim, dropout=0.0)
    mod.eval()

    y = mod(x)

    assert y.shape == x.shape, f"Bad shape: {y.shape}"
    assert torch.isfinite(y).all(), "TorsionResBlock contains non-finite values"

    # residual block should not collapse everything
    diff = (y - x).abs().mean().item()
    assert diff > 0.0, "TorsionResBlock seems to produce no update"

    print("OK: TorsionResBlock")


def test_torsion_head_zero_init_behavior():
    B, L, c_s, n_torsions = 2, 33, 256, 7
    s_initial = torch.randn(B, L, c_s)
    s_final = torch.randn(B, L, c_s)

    mod = TorsionHead(
        c_s=c_s,
        hidden=256,
        n_torsions=n_torsions,
        num_res_blocks=2,
        dropout=0.0
    )
    mod.eval()

    torsions = mod(s_initial, s_final)

    assert torsions.shape == (B, L, n_torsions, 2), f"Bad shape: {torsions.shape}"
    assert torch.isfinite(torsions).all(), "TorsionHead outputs non-finite values"

    # Because final layer is zero-initialized, initial output should be exactly zero
    assert_close(
        torsions,
        torch.zeros_like(torsions),
        atol=1e-7,
        msg="TorsionHead should start near exactly zero because output layer is zero-init"
    )

    print("OK: TorsionHead zero-init behavior")


def test_torsion_head_mask():
    B, L, c_s, n_torsions = 2, 20, 256, 7
    s_initial = torch.randn(B, L, c_s)
    s_final = torch.randn(B, L, c_s)

    mask = torch.ones(B, L)
    mask[0, -4:] = 0.0

    mod = TorsionHead(
        c_s=c_s,
        hidden=256,
        n_torsions=n_torsions,
        num_res_blocks=2,
        dropout=0.0
    )
    mod.eval()

    torsions = mod(s_initial, s_final, mask=mask)

    assert torsions.shape == (B, L, n_torsions, 2), f"Bad shape: {torsions.shape}"
    assert torch.isfinite(torsions).all(), "Masked torsions contain non-finite values"

    assert_close(
        torsions[0, -4:],
        torch.zeros_like(torsions[0, -4:]),
        atol=1e-7,
        msg="Masked torsions should be zero"
    )

    print("OK: TorsionHead mask")


def test_torsion_head_norm_after_breaking_zero_init():
    B, L, c_s, n_torsions = 2, 24, 256, 7
    s_initial = torch.randn(B, L, c_s)
    s_final = torch.randn(B, L, c_s)

    mod = TorsionHead(
        c_s=c_s,
        hidden=256,
        n_torsions=n_torsions,
        num_res_blocks=2,
        dropout=0.0
    )

    # perturb final layer so output is not all zero
    with torch.no_grad():
        mod.output.weight.normal_(mean=0.0, std=0.02)
        mod.output.bias.normal_(mean=0.0, std=0.02)

    mod.eval()
    torsions = mod(s_initial, s_final)

    assert torsions.shape == (B, L, n_torsions, 2)
    assert torch.isfinite(torsions).all(), "TorsionHead outputs non-finite values"

    norms = torch.linalg.norm(torsions, dim=-1)  # [B,L,n_torsions]

    # Only check unit norm where output is actually non-zero numerically
    # after perturbing, almost all should be unit norm
    assert_close(
        norms,
        torch.ones_like(norms),
        atol=1e-4,
        msg="Torsion vectors are not normalized to unit norm"
    )

    print("OK: TorsionHead normalization")


def run_all_head_tests():
    print("Running AlphaFold head tests...\n")
    test_single_projection()
    test_plddt_head()
    test_distogram_head()
    test_torsion_resblock()
    test_torsion_head_zero_init_behavior()
    test_torsion_head_mask()
    test_torsion_head_norm_after_breaking_zero_init()
    print("\nAll head tests passed.")