#!/usr/bin/env python3
"""build_seed_fixture.py — build a synthetic BC checkpoint at the learner's NATIVE obs layout and
convert it to the WEIG folder the C++ learner seeds from.

Runs under the reference venv (Python 3.14) because it imports earl_pytorch (the SAME EARLPerceiver
the BC training used). The C++ .pyd is built against anaconda 3.9 and CANNOT be imported in this
interpreter — that's why seeding is split across two processes: this one only produces the .pt and
the WEIG blobs; smoke_seed.py (anaconda 3.9) loads the .pyd and consumes them.

Native layout (matches TemporalObsBuilder defaults + the existing multi-member smoke test):
  query=59, kv=30, nTokens=559, entityFeatures=30, nControls=9.
"""
import os
import sys

import torch
from torch import nn

from earl_pytorch import EARLPerceiver

N_CONTROLS = 9
N_DIMS = 256
N_HEADS = 8
N_LAYERS = 1
QUERY_FEATURES = 59
KV_FEATURES = 30
N_TOKENS = 559
ENTITY_FEATURES = 30


class ContinuousControlsHead(nn.Module):
    def __init__(self, in_features=N_DIMS, n_controls=N_CONTROLS, init_log_std=-0.5):
        super().__init__()
        self.mean = nn.Linear(in_features, n_controls)
        self.log_std = nn.Parameter(torch.full((n_controls,), float(init_log_std)))

    def forward(self, x):
        mean = self.mean(x)
        log_std = self.log_std.expand(*mean.shape[:-1], -1)
        return torch.cat([mean, log_std], dim=-1)


class PerceiverNet(nn.Module):
    def __init__(self, earl, output, chunk_size=2048):
        super().__init__()
        self.earl = earl
        self.relu = nn.ReLU()
        self.output = output
        self.chunk_size = chunk_size

    def _forward(self, q, kv, m):
        res = self.earl(q, kv, m)
        return self.output(self.relu(res))[:, 0, :]

    def forward(self, inp):
        q, kv, m = inp
        B = q.shape[0]
        if B <= self.chunk_size:
            return self._forward(q, kv, m)
        outs = [self._forward(q[i:i + self.chunk_size], kv[i:i + self.chunk_size],
                             m[i:i + self.chunk_size]) for i in range(0, B, self.chunk_size)]
        return torch.cat(outs, dim=0)


def main():
    out_pt = sys.argv[1]
    torch.manual_seed(1337)

    actor_earl = EARLPerceiver(N_DIMS, N_HEADS, N_LAYERS, 1,
                              query_features=QUERY_FEATURES, key_value_features=KV_FEATURES)
    actor = PerceiverNet(actor_earl, ContinuousControlsHead())
    actor_state = {("net." + k): v for k, v in actor.state_dict().items()}

    critic_earl = EARLPerceiver(N_DIMS, N_HEADS, N_LAYERS, 1,
                              query_features=QUERY_FEATURES, key_value_features=KV_FEATURES)
    critic = PerceiverNet(critic_earl, nn.Linear(N_DIMS, 1))
    critic_state = dict(critic.state_dict())

    os.makedirs(os.path.dirname(os.path.abspath(out_pt)), exist_ok=True)
    torch.save({"actor_state_dict": actor_state, "critic_state_dict": critic_state}, out_pt)
    print(f"[done] wrote synthetic BC checkpoint -> {out_pt}")


if __name__ == "__main__":
    main()
