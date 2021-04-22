from collections import defaultdict
from torch.optim import Optimizer
import torch
import random


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5, k_min=3, k_max=1000):
        print("Using lookahead.")
        self.optimizer = optimizer
        self.resample_k = k <= 0
        self.k_min = k_min
        self.k_max = k_max
        self.k = k if k > 0 else random.randint(k_min, k_max)  # endpoints included
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)
        if self.resample_k:
            self.k = random.randint(self.k_min, self.k_max)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        return loss

    def increment_counter(self):
        for group in self.param_groups:
            group["counter"] += 1

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


def update_avg_gen(G, G_avg, n_gen_update):
    """ Updates the uniform average generator. """
    l_param = list(G.parameters())
    l_avg_param = list(G_avg.parameters())
    if len(l_param) != len(l_avg_param):
        raise ValueError(
            "Got different lengths: {}, {}".format(len(l_param), len(l_avg_param))
        )

    for i in range(len(l_param)):
        with torch.no_grad():
            l_avg_param[i].data.copy_(
                l_avg_param[i]
                .data.mul(n_gen_update)
                .div(n_gen_update + 1.0)
                .add(l_param[i].data.div(n_gen_update + 1.0))
            )


def update_ema_gen(G, G_ema, beta_ema=0.9999):
    """ Updates the exponential moving average generator. """
    l_param = list(G.parameters())
    l_ema_param = list(G_ema.parameters())
    if len(l_param) != len(l_ema_param):
        raise ValueError(
            "Got different lengths: {}, {}".format(len(l_param), len(l_ema_param))
        )

    for i in range(len(l_param)):
        with torch.no_grad():
            l_ema_param[i].data.copy_(
                l_ema_param[i].data.mul(beta_ema).add(l_param[i].data.mul(1 - beta_ema))
            )

