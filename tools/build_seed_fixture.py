#!/usr/bin/env python3
"""build_seed_fixture.py — build a synthetic BC checkpoint at a spec-derived obs layout and convert
it to the WEIG folder the C++ learner seeds from.

Runs under the reference venv (Python 3.14) because it imports earl_pytorch (the SAME EARLPerceiver
the BC training used). The C++ .pyd is built against anaconda 3.9 and CANNOT be imported in this
interpreter — that's why seeding is split across two processes: this one only produces the .pt and
the WEIG blobs; smoke_seed.py / league_driver.py (anaconda 3.9) load the .pyd and consume them.

The fixture records a `spec` dict (profile + n_frames) in the .pt so convert_bc_checkpoint.py can
derive + emit the matching C++ Perceiver widths via spec.json. The actor/critic EARL are built at the
DERIVED widths so the seeded module shape matches what the C++ learner projects to for that member.

Usage: build_seed_fixture.py OUT.pt [--profile on|off] [--n-frames 13]
"""
import argparse
import os
import sys

import torch
from torch import nn

from earl_pytorch import EARLPerceiver

N_CONTROLS = 9
N_DIMS = 256
N_HEADS = 8
N_LAYERS = 1
# Superset layout constants (mirror the reference constants; see convert_bc_checkpoint.spec_to_layout).
_WORLD_FEATURES = 25
_N_PROFILE_FEATURES = 14
_QUERY_TAIL = 169
_RELATIVE_FEATURES = 30
_MAX_TEAM = 4
_MAX_BOOST_PADS = 34


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
    ap = argparse.ArgumentParser(description="build a synthetic BC checkpoint at a spec-derived layout")
    ap.add_argument("out_pt", help="path to write the synthetic BC .pt")
    ap.add_argument("--profile", choices=["on", "off"], default="on",
                    help="profile channel on (39-wide entity) or off (25-wide)")
    ap.add_argument("--n-frames", type=int, default=13, help="temporal buffer depth")
    ap.add_argument("--head", choices=["continuous", "discrete"], default="continuous")
    ap.add_argument("--name", default="fixture", help="bot name recorded in the spec")
    args = ap.parse_args()

    out_pt = args.out_pt
    profile = args.profile == "on"
    n_frames = args.n_frames

    # Derive widths EXACTLY as convert_bc_checkpoint.spec_to_layout does, so the seeded module shape
    # matches what the C++ learner projects to for this member.
    ef = _WORLD_FEATURES + (_N_PROFILE_FEATURES if profile else 0)
    query_features = ef + _QUERY_TAIL
    kv_features = ef + _RELATIVE_FEATURES
    n_tokens = n_frames * (1 + 2 * _MAX_TEAM) + _MAX_BOOST_PADS

    torch.manual_seed(1337)

    actor_earl = EARLPerceiver(N_DIMS, N_HEADS, N_LAYERS, 1,
                              query_features=query_features, key_value_features=kv_features)
    actor = PerceiverNet(actor_earl, ContinuousControlsHead())
    actor_state = {("net." + k): v for k, v in actor.state_dict().items()}

    critic_earl = EARLPerceiver(N_DIMS, N_HEADS, N_LAYERS, 1,
                              query_features=query_features, key_value_features=kv_features)
    critic = PerceiverNet(critic_earl, nn.Linear(N_DIMS, 1))
    critic_state = dict(critic.state_dict())

    # Record the spec so convert_bc_checkpoint.py derives + emits the matching spec.json sidecar.
    spec = {"name": args.name, "profile": profile, "n_frames": n_frames,
            "style_term": None, "alpha": 0.3, "reward": {}, "head": args.head}

    os.makedirs(os.path.dirname(os.path.abspath(out_pt)), exist_ok=True)
    torch.save({"actor_state_dict": actor_state, "critic_state_dict": critic_state, "spec": spec},
               out_pt)
    print(f"[done] wrote synthetic BC checkpoint -> {out_pt} "
          f"(profile={profile} ef={ef} q={query_features} kv={kv_features} tokens={n_tokens})")


if __name__ == "__main__":
    main()
