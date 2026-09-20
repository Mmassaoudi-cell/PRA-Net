"""
Carlini & Wagner (C&W) adversarial-FDIA generation (source paper Sec. II-B,
Eqs. 2-5): given an already BDD-passing FDIA sequence Z^a and a trained
DL detector f, find the minimal-norm perturbation E that flips f's decision
to "non-attacked" while remaining BDD-passing.

Per Eq. 5, E is not a free perturbation in raw measurement space: it must
have the Jacobian-consistent form E_t = H_dc @ c_adv with c_adv supported on
the same (attacked) buses as the original FDIA -- i.e. we optimize directly
over the low-dimensional state-bias vector c_adv, guaranteeing the result
stays exactly BDD-unobservable throughout optimization (the same argument
used to build the base BDD-passing FDIAs in sim/fdia.py). Only the sequence's
final timestep is perturbed (Fig. 1: "assumed to occur only at timestep t").
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def cw_loss(logits, target_class=0):
    """F(Z) = max(l_1 - l_0, 0) if target_class==0 (push toward class 0),
    i.e. the C&W margin loss for forcing the *other* class's logit down."""
    other = 1 - target_class
    return torch.clamp(logits[:, other] - logits[:, target_class], min=0.0)


def craft_adversarial_fdia(model, Z_batch: torch.Tensor, attacked_buses_batch: list[list[int]],
                            H_dc, non_slack: list[int], n_branch: int, n_bus: int,
                            device, C_grid=(1.0, 10.0, 100.0), n_iter=500, lr=0.01):
    """
    Z_batch: (B, k, d) already-BDD-passing FDIA sequences (attack applied at
        the last timestep).
    attacked_buses_batch: list of length B, each a list of attacked bus ids
        (defines the sparsity support for c_adv, per Eq. 5).
    Returns: Z_adv (B,k,d), success mask (B,), l2 norm of E (B,), c_adv (B,n_red)
    """
    B, k, d = Z_batch.shape
    n_red = H_dc.shape[1]
    idx_map = {b: i for i, b in enumerate(non_slack)}
    H_dc_t = torch.tensor(H_dc, dtype=torch.float32, device=device)

    mask = torch.zeros(B, n_red, device=device)
    for i, buses in enumerate(attacked_buses_batch):
        for b in buses:
            if b in idx_map:
                mask[i, idx_map[b]] = 1.0

    Z_batch = Z_batch.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    best_Z = Z_batch.clone()
    best_success = torch.zeros(B, dtype=torch.bool, device=device)
    best_norm = torch.full((B,), float("inf"), device=device)
    best_c = torch.zeros(B, n_red, device=device)

    # cuDNN's RNN backward requires the module to be in train() mode; since
    # the model is frozen (all params require_grad=False) this cannot update
    # any weights/running-stats regardless, so it's safe here. We instead
    # disable cuDNN for this call so eval-mode LSTM/GRU backward works via
    # the generic autograd path without needing train() (which would also
    # make BatchNorm/Dropout stochastic -- undesirable for a deterministic
    # attack).
    with torch.backends.cudnn.flags(enabled=False):
        for C in C_grid:
            c_adv = torch.zeros(B, n_red, device=device, requires_grad=True)
            opt = torch.optim.Adam([c_adv], lr=lr)
            for it in range(n_iter):
                opt.zero_grad()
                c_masked = c_adv * mask
                a = c_masked @ H_dc_t.T  # (B, n_branch+n_bus)
                Z_adv = Z_batch.clone()
                Z_adv[:, -1, :n_branch] = Z_adv[:, -1, :n_branch] + a[:, :n_branch]
                Z_adv[:, -1, 2 * n_branch:2 * n_branch + n_bus] = (
                    Z_adv[:, -1, 2 * n_branch:2 * n_branch + n_bus] + a[:, n_branch:])
                logits = model(Z_adv)
                margin = cw_loss(logits, target_class=0)
                l2 = (a ** 2).sum(-1)
                loss = (l2 + C * margin).sum()
                loss.backward()
                opt.step()

            with torch.no_grad():
                c_masked = c_adv * mask
                a = c_masked @ H_dc_t.T
                Z_adv = Z_batch.clone()
                Z_adv[:, -1, :n_branch] += a[:, :n_branch]
                Z_adv[:, -1, 2 * n_branch:2 * n_branch + n_bus] += a[:, n_branch:]
                logits = model(Z_adv)
                pred = logits.argmax(-1)
                success = pred == 0
                norm = a.norm(dim=-1)
                improve = success & (norm < best_norm)
                best_Z[improve] = Z_adv[improve]
                best_success = best_success | success
                best_norm = torch.where(improve, norm, best_norm)
                best_c[improve] = c_masked[improve]

    return best_Z, best_success, best_norm, best_c
