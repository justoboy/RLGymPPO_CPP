#!/usr/bin/env python3
"""convert_bc_checkpoint.py — Phase 2 BC checkpoint → libtorch converter.

Reads a reference BC checkpoint (a pickled dict with `actor_state_dict` and
`critic_state_dict` sub-dicts, see reference/agent.py) and re-saves each as a
libtorch-loadable zip archive (torch.save of a state_dict), so the C++ learner
can `torch::load(module, stream)` them directly.

Key mapping (the C++ module tree mirrors the Python one EXACTLY):
  actor  : Python keys are `net.earl.*` / `net.output.mean.*` / `net.output.log_std`
           -> strip the leading `net.` -> `earl.*` / `output.mean.*` / `output.log_std`
           (the C++ actor is PerceiverNet{earl, output=ContinuousControlsHead}).
  critic : Python keys are already `earl.*` / `output.weight` / `output.bias`
           (the critic is NOT wrapped in a Policy, so no `net.` prefix) -> pass through.

Round-trip verification: after writing each .lt, we reload it and assert every
tensor matches the source bit-for-bit (sha256 of the float32 bytes), so a silent
dtype/shape drift can never slip through the parity gate.
"""
import argparse
import hashlib
import os
import sys

import torch


def _hash_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _strip_net_prefix(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        if not k.startswith("net."):
            raise ValueError(f"actor key unexpectedly lacks 'net.' prefix: {k}")
        out[k[len("net."):]] = v
    return out


def convert(in_path: str, out_folder: str) -> None:
    if not os.path.isfile(in_path):
        raise FileNotFoundError(in_path)
    os.makedirs(out_folder, exist_ok=True)

    ck = torch.load(in_path, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in ck:
        raise ValueError(
            f"{in_path} is not a BC checkpoint dict "
            f"(top keys: {list(ck.keys())})"
        )

    actor_sd = _strip_net_prefix(ck["actor_state_dict"])
    critic_sd = dict(ck.get("critic_state_dict", {}))  # already un-prefixed; may be absent

    # float32, contiguous, detached — exactly what the C++ module expects.
    actor_sd = {k: v.detach().cpu().to(torch.float32).contiguous() for k, v in actor_sd.items()}
    critic_sd = {k: v.detach().cpu().to(torch.float32).contiguous() for k, v in critic_sd.items()}

    policy_path = os.path.join(out_folder, "PPO_POLICY.lt")
    torch.save(actor_sd, policy_path)
    if critic_sd:
        torch.save(critic_sd, os.path.join(out_folder, "PPO_CRITIC.lt"))

    # Round-trip: reload and hash-compare against the source tensors.
    for label, path, src in (
        ("actor", policy_path, actor_sd),
        ("critic", os.path.join(out_folder, "PPO_CRITIC.lt"), critic_sd),
    ):
        if not src:
            continue
        reloaded = torch.load(path, map_location="cpu", weights_only=False)
        if set(reloaded.keys()) != set(src.keys()):
            raise AssertionError(f"{label} round-trip key mismatch")
        for k in src:
            if _hash_tensor(reloaded[k]) != _hash_tensor(src[k]):
                raise AssertionError(f"{label} round-trip value mismatch at {k}")
        print(f"[ok] {label}: {len(src)} tensors -> {path}")

    print(f"[done] converted {in_path} -> {out_folder}")


def main():
    ap = argparse.ArgumentParser(description="reference BC .pt -> libtorch .lt converter")
    ap.add_argument("in_path", help="path to a reference BC checkpoint .pt")
    ap.add_argument("out_folder", help="folder to write PPO_POLICY.lt / PPO_CRITIC.lt into")
    args = ap.parse_args()
    convert(args.in_path, args.out_folder)


if __name__ == "__main__":
    sys.exit(main())
