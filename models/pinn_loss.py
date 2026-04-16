"""
SimuStruct AI V3 — Physics-Informed Loss (PINN Integration)
=============================================================
Enforces the Navier-Cauchy equilibrium equations as a soft constraint
during training. For static problems with no body forces:

    div(σ) = 0

Expanded in 2D plane stress:
    ∂σ_xx/∂x + ∂σ_xy/∂y = 0
    ∂σ_xy/∂x + ∂σ_yy/∂y = 0

The constitutive law (Hooke's law) relates stress to strain:
    σ = λ·tr(ε)·I + 2μ·ε

where ε = sym(∇u) is the strain tensor and λ, μ are Lamé constants.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def compute_strain_components(
    model: nn.Module,
    coords: torch.Tensor,
    material_feats: torch.Tensor,
    geo_feats: torch.Tensor,
    load_feats: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Compute strain components from displacement predictions using autograd.

    Args:
        model: Neural network predicting [σ_vm, u_x, u_y]
        coords: (N, 2) spatial coordinates (requires_grad=True)
        material_feats: (N, 4) material properties
        geo_feats: (N, 4) geometry parameters
        load_feats: (N, 5) load parameters

    Returns:
        (eps_xx, eps_yy, eps_xy, u, v) strain components and displacements
    """
    coords = coords.requires_grad_(True)

    pred = model(coords, material_feats, geo_feats, load_feats)
    u = pred[:, 1]  # displacement x
    v = pred[:, 2]  # displacement y

    # Compute displacement gradients via autograd
    grad_u = torch.autograd.grad(
        u.sum(), coords, create_graph=True, retain_graph=True
    )[0]
    grad_v = torch.autograd.grad(
        v.sum(), coords, create_graph=True, retain_graph=True
    )[0]

    du_dx = grad_u[:, 0]
    du_dy = grad_u[:, 1]
    dv_dx = grad_v[:, 0]
    dv_dy = grad_v[:, 1]

    # Strain components
    eps_xx = du_dx
    eps_yy = dv_dy
    eps_xy = 0.5 * (du_dy + dv_dx)

    return eps_xx, eps_yy, eps_xy, u, v


def navier_residual(
    model: nn.Module,
    coords: torch.Tensor,
    material_feats: torch.Tensor,
    geo_feats: torch.Tensor,
    lam: float,
    mu: float,
    load_feats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the Navier-Cauchy equilibrium residual.

    For static problems: div(σ) = 0
        ∂σ_xx/∂x + ∂σ_xy/∂y = 0
        ∂σ_xy/∂x + ∂σ_yy/∂y = 0

    Returns:
        Scalar loss value: mean(residual_x² + residual_y²)
    """
    coords = coords.requires_grad_(True)

    # Get strain components
    eps_xx, eps_yy, eps_xy, u, v = compute_strain_components(
        model, coords, material_feats, geo_feats, load_feats
    )

    # Constitutive law (Hooke's law for isotropic material)
    trace = eps_xx + eps_yy
    sig_xx = lam * trace + 2 * mu * eps_xx
    sig_yy = lam * trace + 2 * mu * eps_yy
    sig_xy = 2 * mu * eps_xy

    # Stress divergence (equilibrium residuals)
    dsig_xx_dx = torch.autograd.grad(
        sig_xx.sum(), coords, create_graph=True, retain_graph=True
    )[0][:, 0]

    dsig_xy_dy = torch.autograd.grad(
        sig_xy.sum(), coords, create_graph=True, retain_graph=True
    )[0][:, 1]

    dsig_yy_dy = torch.autograd.grad(
        sig_yy.sum(), coords, create_graph=True, retain_graph=True
    )[0][:, 1]

    dsig_xy_dx = torch.autograd.grad(
        sig_xy.sum(), coords, create_graph=True, retain_graph=True
    )[0][:, 0]

    # Equilibrium: these should be ≈ 0
    res_x = dsig_xx_dx + dsig_xy_dy
    res_y = dsig_xy_dx + dsig_yy_dy

    return (res_x ** 2 + res_y ** 2).mean()


def boundary_loss(
    model: nn.Module,
    boundary_coords: torch.Tensor,
    boundary_values: torch.Tensor,
    material_feats: torch.Tensor,
    geo_feats: torch.Tensor,
    load_feats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Enforce Dirichlet boundary conditions (fixed displacement).

    Args:
        boundary_coords: (M, 2) coordinates of fixed boundary nodes
        boundary_values: (M, 2) prescribed displacement values (usually [0, 0])

    Returns:
        MSE loss on boundary conditions
    """
    pred = model(boundary_coords, material_feats, geo_feats, load_feats)
    pred_disp = pred[:, 1:3]  # displacement components
    return nn.MSELoss()(pred_disp, boundary_values)


def von_mises_consistency_loss(
    pred: torch.Tensor,
    coords: torch.Tensor,
    material_feats: torch.Tensor,
    geo_feats: torch.Tensor,
    model: nn.Module,
    lam: float,
    mu: float,
    load_feats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Enforce consistency between predicted Von Mises stress and the
    stress computed from predicted displacements via constitutive law.

    σ_vm = √(σ_xx² - σ_xx·σ_yy + σ_yy² + 3·σ_xy²)
    """
    coords = coords.requires_grad_(True)

    eps_xx, eps_yy, eps_xy, _, _ = compute_strain_components(
        model, coords, material_feats, geo_feats, load_feats
    )

    trace = eps_xx + eps_yy
    sig_xx = lam * trace + 2 * mu * eps_xx
    sig_yy = lam * trace + 2 * mu * eps_yy
    sig_xy = 2 * mu * eps_xy

    # Von Mises from constitutive law
    vm_computed = torch.sqrt(
        sig_xx ** 2 - sig_xx * sig_yy + sig_yy ** 2 + 3 * sig_xy ** 2 + 1e-8
    )

    # Von Mises from model prediction (channel 0)
    vm_predicted = pred[:, 0]

    return nn.MSELoss()(vm_predicted, vm_computed.detach())


def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    model: nn.Module,
    coords: torch.Tensor,
    material_feats: torch.Tensor,
    geo_feats: torch.Tensor,
    lam: float,
    mu: float,
    load_feats: Optional[torch.Tensor] = None,
    lambda_pde: float = 0.1,
    lambda_bc: float = 0.0,
    boundary_coords: Optional[torch.Tensor] = None,
    boundary_values: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined training loss: data + physics + boundary.

    Args:
        pred: (N, 3) model predictions
        target: (N, 3) ground truth [σ_vm, disp_x, disp_y]
        lambda_pde: Weight for physics loss (default: 0.1)
        lambda_bc: Weight for boundary condition loss

    Returns:
        (total_loss, loss_dict) — total scalar loss and breakdown
    """
    # Data-driven loss
    loss_data = nn.MSELoss()(pred, target)

    loss_dict = {"data": loss_data.item()}
    total = loss_data

    # Physics loss
    if lambda_pde > 0:
        try:
            loss_pde = navier_residual(
                model, coords, material_feats, geo_feats, lam, mu, load_feats
            )
            total = total + lambda_pde * loss_pde
            loss_dict["pde"] = loss_pde.item()
        except Exception:
            loss_dict["pde"] = 0.0

    # Boundary loss
    if lambda_bc > 0 and boundary_coords is not None and boundary_values is not None:
        loss_bc = boundary_loss(
            model, boundary_coords, boundary_values,
            material_feats[:len(boundary_coords)],
            geo_feats[:len(boundary_coords)],
            load_feats[:len(boundary_coords)] if load_feats is not None else None,
        )
        total = total + lambda_bc * loss_bc
        loss_dict["bc"] = loss_bc.item()

    loss_dict["total"] = total.item()
    return total, loss_dict
