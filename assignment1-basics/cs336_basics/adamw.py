import math
import torch
from torch.optim import Optimizer
from typing import Optional, Callable


class AdamW(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate {lr}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"beta1 must be in [0,1), got {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"beta2 must be in [0,1), got {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"eps must >0, got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay >=0, got {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            lam = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # 初始化状态，t从1开始
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    state["t"] = 0
                m = state["m"]
                v = state["v"]
                t = state["t"]
                t += 1

                # -------- Algorithm1 步骤 --------
                # 8: weight decay: θ ← θ − α λ θ
                p.data.mul_(1 - lr * lam)

                # 9: m = β1*m + (1‑β1)*g
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # 10: v = β2*v + (1‑β2)*g²
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 7: corrected learning rate α_t = α * sqrt(1‑β2^t) / (1‑β1^t)
                bias_correction1 = 1.0 - (beta1 ** t)
                bias_correction2 = 1.0 - (beta2 ** t)
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # 11: θ = θ − α_t * m/(sqrt(v)+ε)
                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-alpha_t)

                state["t"] = t
        return loss
