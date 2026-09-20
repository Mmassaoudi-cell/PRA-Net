"""
Candidate 3: EGT-Net/Distill -- a small (m=3) teacher ensemble of plain
(non-evidential) EGTNet backbones is trained independently at train time
only; its ensemble mutual information (epistemic uncertainty, Eq. 10-style)
and averaged per-bus localization probabilities are distilled into a single
student network with the same architecture plus an auxiliary uncertainty-
regression head, so inference requires exactly one forward pass while the
*source* of the uncertainty signal remains true ensemble disagreement
(unlike Candidates 1-2's parametric single-pass mechanisms). See
MODEL_CANDIDATES.md.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from candidates.egt_net import EGTNet, egt_loss, focal_bce
from models.uncertainty import mutual_information


class EGTNetStudent(EGTNet):
    def __init__(self, *args, **kwargs):
        kwargs["use_evidential"] = False
        super().__init__(*args, **kwargs)
        node_dim = self.loc_head.in_features
        self.unc_head = nn.Linear(node_dim, 1)

    def forward_full(self, Z):
        h = self.node_embeddings(Z)
        pooled = h.mean(dim=1)
        det_logits = self.det_head(pooled)
        loc_logits = self.loc_head(h).squeeze(-1)
        u_pred = F.softplus(self.unc_head(pooled)).squeeze(-1)
        return det_logits, loc_logits, u_pred


@torch.no_grad()
def teacher_targets(teachers: list[EGTNet], Z):
    probs_list, loc_probs_list = [], []
    for t in teachers:
        det_out, loc_logits = t(Z)
        probs_list.append(F.softmax(det_out, dim=-1))
        loc_probs_list.append(torch.sigmoid(loc_logits))
    probs_stack = torch.stack(probs_list, 0)  # (m, B, 2)
    u_teacher = mutual_information(probs_stack)  # (B,)
    loc_teacher = torch.stack(loc_probs_list, 0).mean(0)  # (B, n_bus)
    return u_teacher, loc_teacher


def distill_loss(student: EGTNetStudent, teachers: list[EGTNet], Z, y, ybus,
                  kappa1=1.0, kappa2=1.0, gamma_loc=1.0):
    u_teacher, loc_teacher = teacher_targets(teachers, Z)
    det_logits, loc_logits, u_pred = student.forward_full(Z)
    l_det = F.cross_entropy(det_logits, y)
    l_loc_hard = focal_bce(loc_logits, ybus)
    l_loc_soft = F.binary_cross_entropy_with_logits(loc_logits, loc_teacher)
    l_unc = F.mse_loss(u_pred, u_teacher)
    total = l_det + gamma_loc * l_loc_hard + kappa2 * l_loc_soft + kappa1 * l_unc
    return total, dict(l_det=l_det.item(), l_loc_hard=l_loc_hard.item(),
                        l_loc_soft=l_loc_soft.item(), l_unc=l_unc.item())
