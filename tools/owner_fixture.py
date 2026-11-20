#!/usr/bin/env python3
"""owner_fixture.py — P4 active-set owner-map parity gate (Python side + runner).

The per-episode team-config and sibling draws are RNG-driven; Python's random.choices/shuffle and C++'s
std::discrete_distribution/std::mt19937 cannot be made byte-identical, so the per-episode *draw* is not
directly comparable (FEATURE_PORT_PLAN constraint #3: gate on the deterministic transform, not the RNG).
What IS comparable — and what the learner actually routes on — is the deterministic owner-map transform:
given a sampled (n_blue, n_orange), a member count, a sibling/self decision, and a concrete permutation,
the active set's per-slot owner assignment must be identical on both stacks.

Python side mirrors the reference implementation's League.self_matchup (every slot -> one bot_id) and
League.draw_sibling (cycle member ids across slots then permute). The C++ owner_main.cpp mirrors
RLGSC::TeamConfigMutator's owner-map lines. We feed BOTH the same permutation (in place of random.shuffle
/ std::shuffle) so the comparison is deterministic.

The contract being locked: member i has bot_id == i, slot order is blue-first (first n_blue slots blue,
rest orange), self-play => every slot owned by one id, sibling => owner[j] = cycled[perm[j]] where
cycled[j] = j % memberCount. If C++ and Python agree on every case, the env's stamped slotOwners and the
learner's routing agree.

Usage: owner_fixture.py <owner_main.exe>
"""
import random
import subprocess
import sys


def python_owner(member_count: int, sibling: bool, n_blue: int, n_orange: int, perm) -> list:
    """Mirror League.self_matchup / draw_sibling owner semantics.

    Slot order is blue-first (first n_blue slots blue, rest orange). Self-play: every slot -> member 0.
    Sibling: cycle member ids (member i has bot_id i, so cycled[j] = j % member_count) then apply the
    shared permutation out[j] = cycled[perm[j]] in place of random.shuffle."""
    n = n_blue + n_orange
    cycled = [j % member_count for j in range(n)]
    if not sibling:
        return [0] * n
    return [cycled[perm[j]] for j in range(n)]


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: owner_fixture.py <owner_main.exe>")
    exe = sys.argv[1]

    rng = random.Random(1337)
    cases = []      # (member_count, sibling, n_blue, n_orange, perm)
    expected = []   # parallel list of expected owner vectors

    # Sweep member counts and team configs; for sibling cases draw a fresh permutation per case.
    for member_count in (1, 2, 3, 4):
        for (n_blue, n_orange) in ((1, 1), (2, 2), (1, 2), (2, 3), (4, 4)):
            for sibling in (False, True):
                n = n_blue + n_orange
                perm = list(range(n))
                rng.shuffle(perm)
                cases.append((member_count, sibling, n_blue, n_orange, perm))
                expected.append(python_owner(member_count, sibling, n_blue, n_orange, perm))

    # Feed the C++ binary the same cases (perm is the shared permutation in place of std::shuffle).
    stdin_lines = []
    for (member_count, sibling, n_blue, n_orange, perm) in cases:
        stdin_lines.append(
            f"{member_count} {1 if sibling else 0} {n_blue} {n_orange} "
            + " ".join(str(p) for p in perm)
        )
    proc = subprocess.run(
        [exe], input="\n".join(stdin_lines) + "\n",
        text=True, capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"owner_main.exe failed rc={proc.returncode}\n{proc.stderr}")

    cpp_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if len(cpp_lines) != len(expected):
        raise SystemExit(f"case count mismatch: cpp={len(cpp_lines)} py={len(expected)}")

    mismatches = 0
    for i, (line, exp) in enumerate(zip(cpp_lines, expected)):
        got = [int(tok) for tok in line.split()]
        if got != exp:
            mismatches += 1
            print(f"[MISMATCH] case {i}: cpp={got} py={exp}")
    if mismatches:
        print(f"OWNER PARITY FAIL: {mismatches}/{len(expected)} cases diverged")
        sys.exit(1)
    print(f"[done] {len(expected)} owner-map cases match exactly")
    print("OWNER PARITY PASS")


if __name__ == "__main__":
    main()
