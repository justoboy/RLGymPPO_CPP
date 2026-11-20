#!/usr/bin/env python3
"""grad_fixture.py — P3 league gradient-direction parity gate (Python side).

Builds the SAME continuous Gaussian actor the BC checkpoint trained (reference EARLPerceiver +
ContinuousControlsHead), loads the checkpoint's actor_state_dict, then on a seeded flat-obs batch
computes the EXACT PPO policy loss the C++ learner's PPOLearner::Learn runs and backprops it. The
resulting per-parameter gradients are written (keyed by qualified name, sorted) so the C++ GradGate
output can be diffed against them.

This gates the per-member learn path the league routes to: each episode's experience flows to its
owner's buffer and gets this same update. If C++ policy grads match Python's elementwise (cosine ~1.0),
the routing's per-member update math is correct.

Runs under the reference venv (earl_pytorch is py3.14-only). The C++ GradGate.exe (anaconda 3.9 build)
reads the SAME obs.bin/acts.bin/oldlp.bin/adv.bin and writes cpp_keys.txt / cpp_grads.bin.

Writes:
  obs.bin, acts.bin, oldlp.bin, adv.bin  : float32 little-endian fixtures
  meta.txt    : "<N> <obsSize> <q> <kv> <nTokens> <nControls> <entityFeatures> <clipRange> <entCoef>"
  py_keys.txt : one qualified param name per line (sorted)
  py_grads.bin: concatenated flattened grads in py_keys.txt order
"""
import argparse
import os

import numpy as np
import torch
from torch import nn

N_CONTROLS = 9
HALF_LOG_2PI = 0.9189385332046727
HALF_LOG_2PI_E = 1.4189385332046727


class ContinuousControlsHead(nn.Module):
    def __init__(self, in_features=256, n_controls=N_CONTROLS, init_log_std=-0.5):
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


def flat_to_tuple(flat, query_features, kv_features, n_tokens):
    """Rebuild (q, kv, m) from a flat obs row EXACTLY as the C++ PerceiverNet::Forward slices."""
    kv_end = query_features + n_tokens * kv_features
    q = flat[:query_features].reshape(1, 1, query_features)
    kv = flat[query_features:kv_end].reshape(1, n_tokens, kv_features)
    m = flat[kv_end:kv_end + n_tokens].reshape(1, n_tokens)
    return q, kv, m


def main():
    ap = argparse.ArgumentParser(description="P3 gradient-direction parity: emit fixture + Python grads")
    ap.add_argument("checkpoint", help="BC checkpoint .pt (actor_state_dict)")
    ap.add_argument("out_folder", help="folder to write fixtures + py_keys.txt / py_grads.bin")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--query-features", type=int, default=208)
    ap.add_argument("--kv-features", type=int, default=69)
    ap.add_argument("--n-tokens", type=int, default=151)
    ap.add_argument("--entity-features", type=int, default=39)
    ap.add_argument("--n-controls", type=int, default=N_CONTROLS)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    args = ap.parse_args()

    from earl_pytorch import EARLPerceiver

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor_sd = {k[len("net."):]: v for k, v in ck["actor_state_dict"].items()}
    actor_earl = EARLPerceiver(256, 4, 8, 1,
                              query_features=args.query_features,
                              key_value_features=args.kv_features)
    net = PerceiverNet(actor_earl, ContinuousControlsHead(256, args.n_controls))
    net.load_state_dict(actor_sd, strict=True)
    net.train()  # grads need grad enabled

    obs_size = args.query_features + args.n_tokens * args.kv_features + args.n_tokens
    rng = np.random.default_rng(args.seed)
    obs = rng.standard_normal((args.n, obs_size), dtype=np.float32)
    acts = rng.standard_normal((args.n, args.n_controls), dtype=np.float32)
    oldlp = rng.standard_normal((args.n,), dtype=np.float32)
    adv = rng.standard_normal((args.n,), dtype=np.float32)

    tups = [tuple(torch.from_numpy(c).float() for c in
                 flat_to_tuple(obs[i], args.query_features, args.kv_features, args.n_tokens))
            for i in range(args.n)]
    # Stack into a batched (B, 1, q) / (B, nTokens, kv) / (B, nTokens) tuple, matching C++ Forward.
    q = torch.cat([t[0] for t in tups], dim=0)
    kv = torch.cat([t[1] for t in tups], dim=0)
    m = torch.cat([t[2] for t in tups], dim=0)

    acts_t = torch.from_numpy(acts)
    oldlp_t = torch.from_numpy(oldlp)
    adv_t = torch.from_numpy(adv)

    out = net((q, kv, m))  # (N, 2*nControls)
    mean = out[:, :args.n_controls]
    log_std = out[:, args.n_controls:].clamp(-5.0, 2.0)
    std = torch.exp(log_std).clamp(1e-4, 10.0)
    # Independent Normal log_prob summed over channels (matches _channel_log_prob).
    lp = (-0.5 * ((acts_t - mean) / std).pow(2) - torch.log(std) - HALF_LOG_2PI).sum(1)
    ent = (torch.log(std) + HALF_LOG_2PI_E).sum(1)
    entropy = ent.mean()
    ratio = torch.exp(lp - oldlp_t)
    clipped = torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range)
    policy_loss = -torch.min(ratio * adv_t, clipped * adv_t).mean()
    ppo_loss = policy_loss - entropy * args.ent_coef
    net.zero_grad(set_to_none=True)
    ppo_loss.backward()

    # Collect grads keyed by qualified name (net.<...>), sorted, flattened.
    grads = {}
    for name, p in net.named_parameters():
        if p.grad is None:
            raise SystemExit(f"param {name} has no grad")
        grads[name] = p.grad.detach().float().flatten().contiguous()
    keys = sorted(grads.keys())

    os.makedirs(args.out_folder, exist_ok=True)
    obs.astype("<f4").tofile(os.path.join(args.out_folder, "obs.bin"))
    acts.astype("<f4").tofile(os.path.join(args.out_folder, "acts.bin"))
    oldlp.astype("<f4").tofile(os.path.join(args.out_folder, "oldlp.bin"))
    adv.astype("<f4").tofile(os.path.join(args.out_folder, "adv.bin"))
    with open(os.path.join(args.out_folder, "py_keys.txt"), "w") as f:
        f.write("\n".join(keys) + "\n")
    with open(os.path.join(args.out_folder, "py_grads.bin"), "wb") as f:
        for k in keys:
            f.write(grads[k].numpy().tobytes(order="C"))
    with open(os.path.join(args.out_folder, "meta.txt"), "w") as f:
        f.write(f"{args.n} {obs_size} {args.query_features} {args.kv_features} "
                f"{args.n_tokens} {args.n_controls} {args.entity_features} "
                f"{args.clip_range} {args.ent_coef}\n")
    print(f"[done] {args.n} rows obsSize={obs_size} params={len(keys)} -> {args.out_folder}")


if __name__ == "__main__":
    main()
