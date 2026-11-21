#!/usr/bin/env python3
"""reward_fixture.py — GAP1/2/3 reward route+blend parity gate (Python side + runner).

The C++ learner's ComposedReward routes each term's per-slot contribution ONLY to the slots whose owner
selects that term (scaled by that owner's weight), then blends each slot's routed scalar with its teammates'
at the EXPLICIT blue-count boundary, applying that slot's own team_spirit / opponent_punish. The raw term
values come from physics (identical on both stacks), so the only thing that can diverge is this deterministic
route+blend transform. This gate locks it.

Python side mirrors the reference reward.py ComposedReward.get_rewards transform on plain float vectors. The
C++ reward_main.cpp mirrors RLGSC::ComposedReward::GetAllRewards. We feed BOTH the same synthetic per-term raw
vectors / owner map / weights / team_spirit / opponent_punish so the comparison is deterministic (no RNG, no
physics).

Contract locked (identical on both sides):
  routed[j] = sum_i weight[i][owner[j]] * (isDelta[i] ? raw[i][j]-prev[i][j] : raw[i][j])   [owner selects]
  bm = mean(routed[:nBlue]); om = mean(routed[nBlue:])
  blue slot j:  out[j] = (1-ts[owner[j]])*routed[j] + ts[owner[j]]*bm - op[owner[j]]*om
  orange slot j: out[j] = (1-ts[owner[j]])*routed[j] + ts[owner[j]]*om - op[owner[j]]*bm

Usage: reward_fixture.py <reward_main.exe>
"""
import random
import subprocess
import sys

TOL = 1e-6


def python_reward(n, n_blue, member_count, terms, owner, ts, op):
    """Mirror reference ComposedReward.get_rewards on plain vectors.

    terms: list of (isDelta, weights[member_count], raw[n], prev[n])."""
    routed = [0.0] * n
    for (is_delta, weights, raw, prev) in terms:
        for j in range(n):
            contrib = (raw[j] - prev[j]) if is_delta else raw[j]
            o = owner[j]
            if o < 0 or o >= member_count:
                o = 0
            w = weights[o]
            if w != 0.0:
                routed[j] += w * contrib
    bm = sum(routed[:n_blue]) / n_blue if n_blue else 0.0
    om = sum(routed[n_blue:]) / (n - n_blue) if n - n_blue else 0.0
    out = [0.0] * n
    for j in range(n):
        o = owner[j]
        if o < 0 or o >= member_count:
            o = 0
        own = routed[j]
        if j < n_blue:
            out[j] = (1.0 - ts[o]) * own + ts[o] * bm - op[o] * om
        else:
            out[j] = (1.0 - ts[o]) * own + ts[o] * om - op[o] * bm
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: reward_fixture.py <reward_main.exe>")
    exe = sys.argv[1]

    rng = random.Random(2468)
    cases = []  # each: (n, n_blue, member_count, terms, owner, ts, op)

    # Sweep team configs (incl. uneven), member counts, delta/non-delta term mixes, per-member weights.
    for member_count in (1, 2, 3, 4):
        for (n_blue, n_orange) in ((1, 1), (2, 2), (1, 2), (2, 3), (4, 4)):
            for (is_delta, w_mode) in ((0, "uniform"), (1, "permember"), (0, "sparse"), (1, "sparse")):
                n = n_blue + n_orange
                # owner map: blue-first slot order, owner cycles member ids across slots.
                owner = [j % member_count for j in range(n)]
                ts = [rng.uniform(-0.5, 1.0) for _ in range(member_count)]
                op = [rng.uniform(0.0, 1.5) for _ in range(member_count)]
                num_terms = rng.randint(1, 4)
                terms = []
                for _ in range(num_terms):
                    if w_mode == "uniform":
                        w = [rng.uniform(-2.0, 2.0) for _ in range(member_count)]
                    elif w_mode == "permember":
                        base = rng.uniform(-2.0, 2.0)
                        w = [base * (m + 1) for m in range(member_count)]
                    else:  # sparse: only one member selects this term
                        sel = rng.randrange(member_count)
                        w = [0.0] * member_count
                        w[sel] = rng.uniform(-2.0, 2.0)
                    raw = [rng.uniform(-1.0, 1.0) for _ in range(n)]
                    prev = [rng.uniform(-1.0, 1.0) for _ in range(n)] if is_delta else [0.0] * n
                    terms.append((is_delta, w, raw, prev))
                cases.append((n, n_blue, member_count, terms, owner, ts, op))

    # Build the stdin stream for the C++ binary.
    lines = []
    for (n, n_blue, member_count, terms, owner, ts, op) in cases:
        lines.append(f"{n} {n_blue} {member_count} {len(terms)}")
        lines.append(" ".join(str(o) for o in owner))
        lines.append(" ".join(f"{x!r}" for x in ts))
        lines.append(" ".join(f"{x!r}" for x in op))
        for (is_delta, w, raw, prev) in terms:
            parts = [str(is_delta)]
            parts += [f"{x!r}" for x in w]
            parts += [f"{x!r}" for x in raw]
            parts += [f"{x!r}" for x in prev]
            lines.append(" ".join(parts))
        lines.append("")  # blank separator

    proc = subprocess.run(
        [exe], input="\n".join(lines) + "\n",
        text=True, capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"reward_main.exe failed rc={proc.returncode}\n{proc.stderr}")

    cpp_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if len(cpp_lines) != len(cases):
        raise SystemExit(f"case count mismatch: cpp={len(cpp_lines)} py={len(cases)}")

    mismatches = 0
    worst = 0.0
    for i, (line, case) in enumerate(zip(cpp_lines, cases)):
        (n, n_blue, member_count, terms, owner, ts, op) = case
        exp = python_reward(n, n_blue, member_count, terms, owner, ts, op)
        got = [float(tok) for tok in line.split()]
        if len(got) != len(exp):
            print(f"[MISMATCH] case {i}: width cpp={len(got)} py={len(exp)}")
            mismatches += 1
            continue
        for a, b in zip(got, exp):
            d = abs(a - b)
            if d > worst:
                worst = d
            if d > TOL:
                print(f"[MISMATCH] case {i}: cpp={got} py={exp}")
                mismatches += 1
                break
    if mismatches:
        print(f"REWARD PARITY FAIL: {mismatches}/{len(cases)} cases diverged (worst |delta|={worst:g})")
        sys.exit(1)
    print(f"[done] {len(cases)} reward route+blend cases match (max |delta|={worst:g} < {TOL:g})")
    print("REWARD PARITY PASS")


if __name__ == "__main__":
    main()
