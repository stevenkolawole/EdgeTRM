"""Pure-PyTorch AdamATan2, used only where the CUDA extension `adam-atan2`
cannot be built (no nvcc on the node). Same update as the reference
implementation (Everett et al., 2024): the Adam step is replaced by
a * atan2(m_hat, b * sqrt(v_hat)), which needs no epsilon.

Installed as `adam_atan2.py` next to pretrain.py so `from adam_atan2 import
AdamATan2` resolves to this file when the extension is absent.
"""
import math

import torch
from torch.optim.optimizer import Optimizer


class AdamATan2(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0, a=1.27, b=1.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, a=a, b=b)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (beta1, beta2), wd, a, b = (group["lr"], group["betas"], group["weight_decay"],
                                            group["a"], group["b"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                st["step"] += 1
                t = st["step"]
                m, v = st["exp_avg"], st["exp_avg_sq"]
                if wd != 0:
                    p.mul_(1 - lr * wd)
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t
                m_hat = m / bc1
                v_hat = v / bc2
                p.add_(torch.atan2(m_hat, b * v_hat.sqrt()), alpha=-lr * a)
        return loss
