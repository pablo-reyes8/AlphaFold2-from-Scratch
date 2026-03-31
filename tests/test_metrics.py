import torch

torch.manual_seed(0)

def test_structure_metrics():
    B, L = 2, 20
    x_true = torch.randn(B, L, 3)
    mask = torch.ones(B, L)

    # perfect prediction
    x_pred = x_true.clone()
    metrics = compute_structure_metrics(x_pred, x_true, mask, align=True)

    print("Perfect prediction:")
    print("RMSD:", metrics["rmsd"].item())
    print("TM-score:", metrics["tm_score"].item())
    print("GDT-TS:", metrics["gdt_ts"].item())

    assert metrics["rmsd"].item() < 1e-3, "RMSD should be ~0"
    assert metrics["tm_score"].item() > 0.999, "TM-score should be ~1"
    assert metrics["gdt_ts"].item() > 0.999, "GDT-TS should be ~1"

    # perturbed prediction
    x_pred2 = x_true + 0.8 * torch.randn_like(x_true)
    metrics2 = compute_structure_metrics(x_pred2, x_true, mask, align=True)

    print("\nPerturbed prediction:")
    print("RMSD:", metrics2["rmsd"].item())
    print("TM-score:", metrics2["tm_score"].item())
    print("GDT-TS:", metrics2["gdt_ts"].item())

    assert metrics2["rmsd"].item() > metrics["rmsd"].item(), "RMSD should worsen"
    assert metrics2["tm_score"].item() < metrics["tm_score"].item(), "TM-score should worsen"
    assert metrics2["gdt_ts"].item() < metrics["gdt_ts"].item(), "GDT-TS should worsen"

    print("\nAll metric tests passed.")

test_structure_metrics()