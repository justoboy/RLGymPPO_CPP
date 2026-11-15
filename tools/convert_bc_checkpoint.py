#!/usr/bin/env python3
"""convert_bc_checkpoint.py — Phase 2 BC checkpoint -> portable binary state-dict converter.

Reads a reference BC checkpoint (a pickled dict with `actor_state_dict` and
`critic_state_dict` sub-dicts, see reference/agent.py) and writes each as a tiny
self-describing BINARY blob (see RLGPC::LoadStateDict in StateDictIO.h).

Why not just torch.save the state_dict? Because torch's pickle archive format is NOT
stable across versions: a .pt written by the BC training env (torch 2.14) embeds
`constants.pkl` that the C++ libtorch 2.2.0 reader rejects. A raw binary blob sidesteps
the whole cross-version serialization problem — the C++ side just memcpy's float32
buffers into the module by qualified name.

Key mapping (the C++ module tree mirrors the Python one EXACTLY):
  actor  : Python keys are `net.earl.*` / `net.output.mean.*` / `net.output.log_std`
           -> strip the leading `net.` -> `earl.*` / `output.mean.*` / `output.log_std`
           (the C++ actor is PerceiverNet{earl, output=ContinuousControlsHead}).
  critic : Python keys are already `earl.*` / `output.weight` / `output.bias`
           (the critic is NOT wrapped in a Policy, so no `net.` prefix) -> pass through.

Round-trip verification: after writing each blob we reload it and assert every tensor
matches the source bit-for-bit (sha256 of the float32 bytes).
"""
import argparse
import hashlib
import os
import struct
import sys

import torch

MAGIC = 0x57454947  # "WEIG"
VERSION = 1


def _hash_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _strip_net_prefix(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        if not k.startswith("net."):
            raise ValueError(f"actor key unexpectedly lacks 'net.' prefix: {k}")
        out[k[len("net."):]] = v
    return out


def _write_blob(path: str, state_dict: dict) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<III", MAGIC, VERSION, len(state_dict)))
        for k, v in state_dict.items():
            v = v.detach().cpu().contiguous().to(torch.float32)
            kb = k.encode("utf-8")
            f.write(struct.pack("<I", len(kb)))
            f.write(kb)
            dims = list(v.shape)
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<I", int(d)))
            f.write(v.numpy().tobytes(order="C"))


def _read_blob(path: str) -> dict:
    with open(path, "rb") as f:
        magic, version, num = struct.unpack("<III", f.read(12))
        if magic != MAGIC:
            raise AssertionError("bad magic on reload")
        if version != VERSION:
            raise AssertionError("bad version on reload")
        out = {}
        for _ in range(num):
            (klen,) = struct.unpack("<I", f.read(4))
            key = f.read(klen).decode("utf-8")
            (rank,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<I", f.read(4))[0] for _ in range(rank)]
            numel = 1
            for d in dims:
                numel *= d
            raw = f.read(numel * 4)
            arr = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(dims)
            out[key] = arr
        return out


def convert(in_path: str, out_folder: str) -> None:
    if not os.path.isfile(in_path):
        raise FileNotFoundError(in_path)
    os.makedirs(out_folder, exist_ok=True)

    ck = torch.load(in_path, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in ck:
        raise ValueError(
            f"{in_path} is not a BC checkpoint dict (top keys: {list(ck.keys())})"
        )

    actor_sd = _strip_net_prefix(ck["actor_state_dict"])
    critic_sd = dict(ck.get("critic_state_dict", {}))  # already un-prefixed; may be absent
    # FSP head state_dict is already un-prefixed (earl.* / head.weight / head.bias); the C++ FSPNet
    # registers the SAME shared earl trunk plus a `head` Linear, so keys map 1:1. Absent when the
    # source checkpoint predates FSP (single-bot BC->RL), in which case we emit no FSP blob.
    fsp_sd = dict(ck.get("fsp_state_dict", {}))

    policy_path = os.path.join(out_folder, "PPO_POLICY.lt")
    _write_blob(policy_path, actor_sd)
    critic_path = os.path.join(out_folder, "PPO_CRITIC.lt")
    if critic_sd:
        _write_blob(critic_path, critic_sd)
    fsp_path = os.path.join(out_folder, "PPO_FSP.lt")
    if fsp_sd:
        _write_blob(fsp_path, fsp_sd)

    # Round-trip: reload and hash-compare against the source tensors.
    for label, path, src in (
        ("actor", policy_path, actor_sd),
        ("critic", critic_path, critic_sd),
        ("fsp", fsp_path, fsp_sd),
    ):
        if not src:
            continue
        reloaded = _read_blob(path)
        if set(reloaded.keys()) != set(src.keys()):
            raise AssertionError(f"{label} round-trip key mismatch")
        for k in src:
            if _hash_tensor(reloaded[k]) != _hash_tensor(src[k]):
                raise AssertionError(f"{label} round-trip value mismatch at {k}")
        print(f"[ok] {label}: {len(src)} tensors -> {path}")

    print(f"[done] converted {in_path} -> {out_folder}")


def main():
    ap = argparse.ArgumentParser(description="reference BC .pt -> portable binary state-dict converter")
    ap.add_argument("in_path", help="path to a reference BC checkpoint .pt")
    ap.add_argument("out_folder", help="folder to write PPO_POLICY.lt / PPO_CRITIC.lt into")
    args = ap.parse_args()
    convert(args.in_path, args.out_folder)


if __name__ == "__main__":
    sys.exit(main())
