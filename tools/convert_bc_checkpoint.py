#!/usr/bin/env python3
"""convert_bc_checkpoint.py — Phase 2 BC checkpoint -> portable binary state-dict converter.

Reads a reference BC checkpoint (a pickled dict with `actor_state_dict` and
`critic_state_dict` sub-dicts, see the reference agent module) and writes each as a tiny
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
import json
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


# --- spec -> C++ PerceiverConfig width derivation ---------------------------------------------
# Mirrors the reference bot_spec + constants so the (py3.9) driver can build a matching
# PerceiverConfig WITHOUT importing the reference impl (earl_pytorch is py3.14-only). These literals
# MUST stay in sync with the reference constants: WORLD_FEATURES=25, N_PROFILE_FEATURES=14, QUERY_TAIL=161,
# RELATIVE_FEATURES=25, MAX_TEAM=4 (=> 2*MAX_TEAM=8 car slots), MAX_BOOST_PADS=34, N_TEMPORAL_FRAMES=13.
# v2: query = q_self(entity_features - 5, the non-IS_* cols) + tail(161: ball6+tactical3+mode-fields10+
# tiles140+team2); the mode is a trailing scalar channel, not a query column.
_WORLD_FEATURES = 25
_N_PROFILE_FEATURES = 14
_QUERY_TAIL = 161
_RELATIVE_FEATURES = 25
_MAX_TEAM = 4
_MAX_BOOST_PADS = 34
_N_CONTROLS = 9


def spec_to_layout(spec_dict: dict) -> dict:
    """Derive the C++ PerceiverConfig widths from a checkpoint's recorded BotSpec dict.

    entity_features = 25 + (14 if profile else 0)
    query_features  = (entity_features - 5) + QUERY_TAIL(161)   # q_self (non-IS_*) + tail
    kv_features     = entity_features + RELATIVE_FEATURES(25)
    n_tokens        = n_frames * (1 + 2*MAX_TEAM) + MAX_BOOST_PADS  (13*9 + 34 = 151)
    head_type       = 'continuous' | 'discrete' (from spec.head)
    """
    profile = bool(spec_dict.get("profile", True))
    n_frames = int(spec_dict.get("n_frames", 13))
    ef = _WORLD_FEATURES + (_N_PROFILE_FEATURES if profile else 0)
    return {
        "entity_features": ef,
        "query_features": (ef - 5) + _QUERY_TAIL,
        "kv_features": ef + _RELATIVE_FEATURES,
        "n_tokens": n_frames * (1 + 2 * _MAX_TEAM) + _MAX_BOOST_PADS,
        "n_controls": _N_CONTROLS,
        "head": str(spec_dict.get("head", "continuous")),
        "profile": profile,
        "n_frames": n_frames,
    }


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

    # Sidecar: if the checkpoint recorded its BotSpec, derive + emit the C++ PerceiverConfig widths so
    # the (py3.9) driver can build a matching learner without importing the reference impl. Absent spec -> no
    # sidecar (the driver falls back to a caller-supplied layout, e.g. the native smoke layout).
    spec_dict = ck.get("spec")
    if isinstance(spec_dict, dict):
        layout = spec_to_layout(spec_dict)
        with open(os.path.join(out_folder, "spec.json"), "w", encoding="utf-8") as f:
            json.dump(layout, f, indent=2, sort_keys=True)
        print(f"[ok] spec.json -> {layout}")
    else:
        print("[info] checkpoint has no recorded 'spec'; no spec.json sidecar emitted")


def main():
    ap = argparse.ArgumentParser(description="reference BC .pt -> portable binary state-dict converter")
    ap.add_argument("in_path", help="path to a reference BC checkpoint .pt")
    ap.add_argument("out_folder", help="folder to write PPO_POLICY.lt / PPO_CRITIC.lt into")
    args = ap.parse_args()
    convert(args.in_path, args.out_folder)


if __name__ == "__main__":
    sys.exit(main())
