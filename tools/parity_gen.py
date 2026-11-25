#!/usr/bin/env python3
"""parity_gen.py — Phase 2 parity gate: generate seeded obs batches + Python action means.

Builds the SAME continuous Gaussian policy the BC checkpoint trained (the reference agent's
EARLPerceiver + ContinuousControlsHead), loads the checkpoint's actor_state_dict into it,
then for N seeded FLAT obs vectors (q||kv||m, obsSize wide) computes the deterministic
policy output (the raw 9 control means, pre-decode).

Writes a binary manifest the C++ parity executable reads:
  meta.txt          : "<N> <obsSize> <queryFeatures> <kvFeatures> <nTokens> <nControls>"
  obs.bin           : float32 little-endian (N, obsSize)  (C-contiguous row-major)
  py_means.bin      : float32 little-endian (N, nControls) Python deterministic means

The C++ side rebuilds the (q, kv, m) tuple from the SAME flat obs (identical slicing) and
compares its GaussianPolicy deterministic means against py_means. max|delta| < 1e-4 => PASS.
"""
import argparse
import os

import numpy as np
import torch
from torch import nn

from earl_pytorch import EARLPerceiver

# --- Mirror of the reference agent (kept local so the gate has zero reference-import deps) ---
N_CONTROLS = 9
_N_MODES = 7


class ModeAwareEARL(EARLPerceiver):
    """Library EARLPerceiver + a 7-way mode embedding added to the query embedding (v2).

    Mirrors the reference perceiver: subclass the (unmodifiable) library EARLPerceiver, add
    self.mode_emb = nn.Embedding(N_MODES, n_dims), override forward to inject the mode into the
    query embedding. State-dict key = mode_emb.weight (under the earl.* namespace once registered
    as PerceiverNet.earl), matching the C++ EARLPerceiverImpl.
    """

    def __init__(self, n_dims, n_layers, n_heads, n_preprocess_layers, query_features, key_value_features):
        super().__init__(n_dims, n_layers, n_heads, n_preprocess_layers,
                         query_features=query_features, key_value_features=key_value_features)
        self.mode_emb = nn.Embedding(_N_MODES, n_dims)

    def forward(self, q, kv, m, mode):
        q_emb = self.query_preprocess(q)
        kv_emb = self.key_value_preprocess(kv)
        q_emb = q_emb + self.mode_emb(mode.reshape(-1).long()).unsqueeze(1)
        for block in self.blocks:
            q_emb, _ = block(q_emb, kv_emb, m)
        return q_emb


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

    def _forward(self, q, kv, m, mode):
        res = self.earl(q, kv, m, mode)
        res = self.output(self.relu(res))
        return res[:, 0, :]

    def forward(self, inp):
        q, kv, m, mode = inp
        B = q.shape[0]
        if B <= self.chunk_size:
            return self._forward(q, kv, m, mode)
        outs = [self._forward(q[i:i + self.chunk_size], kv[i:i + self.chunk_size],
                             m[i:i + self.chunk_size], mode[i:i + self.chunk_size])
                for i in range(0, B, self.chunk_size)]
        return torch.cat(outs, dim=0)


class FSPNet(nn.Module):
    """Mirror of the reference FSPNet. Holds the SAME earl trunk instance the actor uses, plus a
    Linear(256, out_features) head. forward((q,kv,m,mode)) -> (B, out_features) = head(relu(earl))[:,0,:]."""

    def __init__(self, earl, out_features, chunk_size=2048):
        super().__init__()
        self.earl = earl
        self.relu = nn.ReLU()
        self.head = nn.Linear(256, out_features)
        self.chunk_size = chunk_size

    def _forward(self, q, kv, m, mode):
        res = self.earl(q, kv, m, mode)
        return self.head(self.relu(res))[:, 0, :]

    def forward(self, inp):
        q, kv, m, mode = inp
        B = q.shape[0]
        if B <= self.chunk_size:
            return self._forward(q, kv, m, mode)
        outs = [self._forward(q[i:i + self.chunk_size], kv[i:i + self.chunk_size],
                             m[i:i + self.chunk_size], mode[i:i + self.chunk_size])
                for i in range(0, B, self.chunk_size)]
        return torch.cat(outs, dim=0)


def flat_to_tuple(flat, query_features, kv_features, n_tokens):
    """Rebuild (q, kv, m, mode) from a flat obs row EXACTLY as the C++ PerceiverNet::Forward slices.
    v2: mode is the single trailing float column after the mask block; the model casts it to int64."""
    kv_end = query_features + n_tokens * kv_features
    mask_end = kv_end + n_tokens
    q = flat[:query_features].reshape(1, 1, query_features)
    kv = flat[query_features:kv_end].reshape(1, n_tokens, kv_features)
    m = flat[kv_end:mask_end].reshape(1, n_tokens)
    mode = flat[mask_end:mask_end + 1]
    return q, kv, m, mode


def main():
    ap = argparse.ArgumentParser(description="Phase 2 parity: emit seeded obs + Python action means")
    ap.add_argument("checkpoint", help="BC checkpoint .pt (actor_state_dict and/or fsp_state_dict)")
    ap.add_argument("out_folder", help="folder to write obs.bin / py_means.bin / py_fsp.bin / meta.txt")
    ap.add_argument("--n", type=int, default=64, help="number of seeded obs rows")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--query-features", type=int, default=195)
    ap.add_argument("--kv-features", type=int, default=64)
    ap.add_argument("--n-tokens", type=int, default=151)
    ap.add_argument("--entity-features", type=int, default=39)
    ap.add_argument("--n-controls", type=int, default=N_CONTROLS)
    ap.add_argument("--actor", action="store_true", help="emit continuous actor (py_means.bin) fixture")
    ap.add_argument("--fsp", action="store_true", help="emit FSP head (py_fsp.bin) fixture")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # The actor and FSP paths are INDEPENDENTLY gated. The continuous BC checkpoint carries only an
    # actor_state_dict; the league checkpoint carries an fsp_state_dict (and a discrete actor we don't
    # decode here). Each path builds its OWN mode-aware earl with the widths baked into its weights.
    net = None
    if args.actor:
        # The actor is wrapped in a Policy (self.net = PerceiverNet), so its keys carry a
        # leading "net." prefix; strip it to match the bare PerceiverNet built here.
        actor_sd = {k[len("net."):]: v for k, v in ck["actor_state_dict"].items()}
        actor_earl = ModeAwareEARL(256, 4, 8, 1,
                                  query_features=args.query_features,
                                  key_value_features=args.kv_features)
        head = ContinuousControlsHead(256, args.n_controls)
        net = PerceiverNet(actor_earl, head)
        net.load_state_dict(actor_sd, strict=True)
        net.eval()

    fsp = None
    if args.fsp:
        if "fsp_state_dict" not in ck:
            raise SystemExit(f"{args.checkpoint} has no fsp_state_dict; cannot emit FSP fixture")
        fsp_earl = ModeAwareEARL(256, 4, 8, 1,
                               query_features=args.query_features,
                               key_value_features=args.kv_features)
        # v2: FSP head predicts the self row's NON-IS_* cols (q_self = entity_features - 5 wide).
        fsp = FSPNet(fsp_earl, args.entity_features - 5)
        fsp.load_state_dict(ck["fsp_state_dict"], strict=True)
        fsp.eval()

    # v2: mode is a trailing scalar channel, so obsSize gains +1 over the q++kv++m width.
    obs_size = args.query_features + args.n_tokens * args.kv_features + args.n_tokens + 1
    rng = np.random.default_rng(args.seed)
    obs = rng.standard_normal((args.n, obs_size), dtype=np.float32)

    q_self = args.entity_features - 5
    means = np.zeros((args.n, args.n_controls), dtype=np.float32)
    fsp_out = np.zeros((args.n, q_self), dtype=np.float32)
    with torch.no_grad():
        for i in range(args.n):
            tup = tuple(torch.from_numpy(c).float() for c in
                        flat_to_tuple(obs[i], args.query_features, args.kv_features, args.n_tokens))
            if net is not None:
                out = net(tup)                   # (1, 2*nControls)
                means[i] = out[0, :args.n_controls].numpy()
            if fsp is not None:
                fsp_out[i] = fsp(tup).numpy()    # (1, entity_features - 5)

    os.makedirs(args.out_folder, exist_ok=True)
    obs.astype("<f4").tofile(os.path.join(args.out_folder, "obs.bin"))
    if net is not None:
        means.astype("<f4").tofile(os.path.join(args.out_folder, "py_means.bin"))
    if fsp is not None:
        fsp_out.astype("<f4").tofile(os.path.join(args.out_folder, "py_fsp.bin"))
    with open(os.path.join(args.out_folder, "meta.txt"), "w") as f:
        f.write(f"{args.n} {obs_size} {args.query_features} {args.kv_features} "
                f"{args.n_tokens} {args.n_controls} {args.entity_features} "
                f"{1 if net is not None else 0} {1 if fsp is not None else 0}\n")

    print(f"[done] {args.n} rows obsSize={obs_size} actor={net is not None} fsp={fsp is not None} "
          f"-> {args.out_folder}")


if __name__ == "__main__":
    main()
