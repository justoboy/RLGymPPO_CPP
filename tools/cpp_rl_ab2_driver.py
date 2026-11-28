"""C++-learner curriculum driver for the rl_ab2 BC-vs-random / discrete-vs-continuous ablation.

Runs under anaconda 3.9 (it imports the cp39 rlgym_learner .pyd). ONE invocation trains ONE (arm, seed)
run: a single-member learner (every slot owned by member 0 = self-play) whose head type + init come from
the job, driven through a CONFIG-DRIVEN SEQUENTIAL PHASE curriculum. The C++ side is dumb (it just spawns
the first enabled DrillSpec per episode); ALL curriculum intelligence lives here in the job JSON + this
interpreter, so tuning the curriculum never requires code changes.

JOB SCHEMA (v2) — the curriculum is a list of ordered phases; exactly one is ACTIVE at a time (the first
whose state != OFF). Each phase:

    {
      "name":   "touch",                     # label (curve + state file)
      "metric": "touches_for_mean",          # report key the state machine advances on
      "x": 1.0, "y": 5, "z": 5,              # hit bar / gate streak / fade score target
      "smooth_window": 0,                    # optional: FADE walk decides on rolling mean of last N
      "reward": { ... },                     # optional reward program while this phase is active
                                             #   (absent => job.default_reward)
      "drill": {                             # the DrillSpec this phase pins onto the env
        "spawn_type": "TOUCH",               #   TOUCH | GOAL_BREAKAWAY | NATIVE_KICKOFF
        "n_blue": 1, "n_orange": 0,          #   lineup override (omit => sampled lineup)
        "car_dist": [400, 4000],             #   scalar, or [easy, hard] interpolated by g
        "ball_y":   [5120, 0],               #   scalar, or [easy, hard] interpolated by g
        "clock_cap": 10.0,                   #   per-episode clock cap (s)
        "hard_finish": false                 #   capped NATIVE_KICKOFF ends on expiry (no OT)
      }
    }

Phase state machine (FADE-FIRST-THEN-GATE, same walk as before, but the GATE now measures on the
phase's OWN drill pinned at MAX difficulty (g=1) — no native fallback episode):

    FADE: score = clamp(score + (metric >= x ? +2 : -3), 0, z); score >= z -> GATE (streak = 0)
          The penalty (-3) EXCEEDS the reward (+2): the walk climbs only above the ~60% break-even, so a
          lucky epoch cannot carry a phase forward. g = score/z interpolates the drill geometry from
          easy (g=0) to hard (g=1).
    GATE: drill stays enabled at g=1 (hardest); streak = streak+1 if RAW metric >= x else 0;
          streak >= y -> OFF. Retirement therefore proves competence at the hardest drill, not native.
    OFF:  phase retired; the NEXT phase (if any) unlocks to FADE on the next epoch.

ALWAYS-DRILL: during a drill phase every episode spawns the phase's drill (no fire probability, no
native mixing — mixing native spawns into drill phases was observed to hurt learning). The only escape
hatch is the job-level "native_fraction" (default 0.0), which routes a fraction of episodes to a plain
native kickoff.

When ALL phases are OFF the run plays native full matches under job.default_reward.

RESUME: <out_path>.state.json (schema_version 2) restores phase states + done_epochs + curve; the
learner Load()s weights from load_dir (no re-seed). A v1 (pre-drill-rework) state file is REJECTED —
the curriculum semantics changed incompatibly, so delete it and restart the run.
"""
import json
import os
import sys

# Reuse the league driver's helpers (same tools/ dir): the .pyd import dance, the BC .pt -> WEIG converter
# shell-back, the spec.json reader, and the EnvConfig/PerceiverConfig builders. Single source of truth.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from league_driver import (  # noqa: E402
    _import_module, convert_checkpoint, read_spec, build_env_config, build_perceiver, NATIVE,
)

STATE_SCHEMA_VERSION = 2


def split_reward(rmap):
    """Split a reward map into (term_weights_without_reserved, team_spirit, opponent_punish). The reserved
    keys team_spirit/opponent_punish are passed to set_member_reward as scalars, mirroring the reference."""
    m = dict(rmap)
    ts = float(m.pop("team_spirit", 0.6))
    op = float(m.pop("opponent_punish", 1.0))
    return m, ts, op


def _interp(rng, g):
    """Geometry value: scalar => fixed; [easy, hard] => linear interpolation by the fade fraction g."""
    if isinstance(rng, (list, tuple)):
        return float(rng[0]) + (float(rng[1]) - float(rng[0])) * g
    return float(rng)


def advance_phase(p, metric, smooth_window):
    """Advance one phase's state machine by one epoch given its epoch-mean metric (FADE-FIRST-THEN-GATE).
    FADE: score = clamp(score + (+2 if hit else -3), 0, z); score >= z -> GATE (streak = 0).
    GATE: streak = streak+1 if RAW metric >= x else 0; streak >= y -> OFF. No-op once OFF or PENDING.

    SMOOTHING (smooth_window > 0): the FADE walk's `hit` is decided on the ROLLING MEAN of the last
    `smooth_window` epoch-metrics, not the single raw epoch. This collapses the per-epoch sampling
    variance that otherwise turns the +2/-3 walk into a diffusion-dominated random walk absorbed at the
    0 floor. The GATE proof stays on the RAW metric (a clean hardest-drill competence test, never
    smoothed). The rolling buffer lives in p["hist"] (persisted)."""
    state = p["state"]
    if state in ("OFF", "PENDING"):
        return p
    hist = list(p.get("hist", []))
    if smooth_window > 0:
        hist = (hist + [float(metric)])[-smooth_window:]
        hit = (sum(hist) / len(hist)) >= p["x"] if hist else (metric >= p["x"])
    else:
        hit = metric >= p["x"]
    if state == "FADE":
        score = max(0.0, min(float(p["z"]), p["score"] + (2.0 if hit else -3.0)))
        if score >= p["z"]:
            return {**p, "state": "GATE", "streak": 0, "score": float(p["z"]), "hist": []}
        return {**p, "state": "FADE", "streak": 0, "score": score, "hist": hist}
    # GATE: the drill stays enabled at g=1 (hardest geometry); the retirement proof is the RAW metric.
    raw_hit = metric >= p["x"]
    streak = p["streak"] + 1 if raw_hit else 0
    if streak >= p["y"]:
        return {**p, "state": "OFF", "streak": 0, "hist": []}
    return {**p, "state": "GATE", "streak": streak, "hist": hist}


def phase_geometry(p):
    """(firing, g) for a phase: firing = the drill spawns (FADE or GATE — always-drill); g = the fade
    fraction (score/z during FADE, pinned to 1.0 during GATE = hardest drill)."""
    firing = p["state"] in ("FADE", "GATE")
    g = 1.0 if p["state"] == "GATE" else ((p["score"] / p["z"]) if p["z"] > 0 else 0.0)
    return firing, g


def active_phase(phases):
    """The first phase whose state != OFF (None => all retired => native play)."""
    for p in phases:
        if p["state"] != "OFF":
            return p
    return None


def build_drill_specs(L, phases):
    """Resolve the phase list into the ordered DrillSpec list the C++ mutator evaluates per episode:
    exactly the ACTIVE phase's drill is enabled (always-drill, first-enabled-wins), everything else off."""
    specs = []
    act = active_phase(phases)
    for p in phases:
        d = p["drill"]
        firing, g = phase_geometry(p)
        enabled = bool(act is not None and p is act and firing)
        s = L.DrillSpec()
        s.spawn_type = getattr(L.DrillSpawnType, str(d["spawn_type"]))
        s.enabled = enabled
        s.n_blue = int(d.get("n_blue", -1))
        s.n_orange = int(d.get("n_orange", -1))
        s.car_dist = _interp(d.get("car_dist", 400.0), g)
        s.ball_y = _interp(d.get("ball_y", 0.0), g)
        s.clock_cap = float(d.get("clock_cap", -1.0))
        s.hard_finish = bool(d.get("hard_finish", False))
        specs.append(s)
    return specs


def phase_reward(phases, job):
    """Reward program (a STEP, no blending): the ACTIVE phase's own reward map if it carries one, else
    job.default_reward. All phases OFF => default_reward (native play)."""
    act = active_phase(phases)
    if act is not None and act.get("reward"):
        return dict(act["reward"])
    return dict(job["default_reward"])


def build_learner(L, job, layout):
    """Single-member learner (every slot owned by member 0 = self-play). head from job.head / layout."""
    init_rmap = job["phases"][0].get("reward") if bool(job.get("curriculum", True)) else None
    init_rmap = init_rmap or job["default_reward"]
    init_terms, init_ts, init_op = split_reward(init_rmap)
    mrc = L.MemberRewardConfig()
    mrc.termWeights = [(str(k), float(v)) for k, v in init_terms.items()]
    mrc.teamSpirit = float(init_ts)
    mrc.opponentPunish = float(init_op)

    ec = build_env_config(
        L, [layout], tick_skip=int(job["tick_skip"]),
        game_mode=getattr(L.GameMode, str(job["game_mode"])),
        team_configs=[(int(job["n_blue"]), int(job["n_orange"]))], team_weights=[1.0],
        member_count=1, sibling_fraction=0.0, random_seed=int(job["seed"]),
        member_reward_configs=[{"term_weights": init_terms, "team_spirit": init_ts,
                                "opponent_punish": init_op}],
        native_time=float(job["native_time"]),
        target_goals=int(job.get("target_goals", 1)),
        max_overtime_seconds=float(job["max_overtime_seconds"]),
        # Native matches run the full native_time + OT (kickoff respawn on goal, no no-touch truncation),
        # matching the reference's rl_ab2 behavior. Drill episodes cap at their own clock_cap; the duel
        # (a NATIVE_KICKOFF spec with clock_cap + hard_finish) ends on expiry with no OT.
        full_match=bool(job.get("full_match", True)))

    pc = build_perceiver(L, layout)
    cfg = L.LearnerConfig()
    cfg.deviceType = getattr(L.LearnerDeviceType, str(job["device"]))
    cfg.collisionMeshesFolder = os.environ.get(
        "COLLISION_MESHES", r"D:\RLBotTraining\RLGymSim_CPP\collision_meshes")
    cfg.numThreads = int(job["num_threads"])
    cfg.numGamesPerThread = int(job["games_per_thread"])
    cfg.timestepsPerIteration = int(job["timesteps_per_iteration"])
    cfg.expBufferSize = int(job["exp_buffer_size"])
    # Match-based collection (the reference implementation's curriculum driver): when matches_per_epoch
    # > 0, one iteration collects that many COMPLETED episodes (drills cut short by their time cap still
    # count, since a time-capped drill emits done). timesteps_per_iteration then only guards against a
    # hung match. 0 (or absent) keeps the legacy pure step-target collection.
    cfg.matchesPerIteration = int(job.get("matches_per_epoch", 0))
    cfg.standardizeReturns = bool(job["standardize_returns"])
    cfg.checkpointSaveFolder = str(job["save_dir"])
    cfg.checkpointLoadFolder = str(job.get("load_dir") or "")
    cfg.sendMetrics = False
    cfg.randomSeed = int(job["seed"])
    cfg.ppo.batchSize = int(job["batch_size"])
    cfg.ppo.miniBatchSize = int(job["mini_batch_size"])
    cfg.ppo.epochs = int(job["epochs_ppo"])
    cfg.ppo.policyLR = float(job["actor_lr"])
    cfg.ppo.criticLR = float(job["critic_lr"])
    cfg.ppo.entCoef = float(job["ent_coef"])
    cfg.ppo.clipRange = float(job.get("clip_range", 0.2))
    cfg.ppo.lambdaFSP = float(job["lambda_fsp"])
    cfg.memberPerceiver = [pc]
    return L.create_learner(ec, cfg)


def _new_phase(name, metric, x, y, z, reward, drill, state):
    return {"name": name, "metric": metric, "state": state, "streak": 0, "score": 0.0,
            "x": float(x), "y": int(y), "z": float(z), "hist": [],
            "reward": reward, "drill": drill}


def fresh_phases(job):
    """Build the phase state list from the job. curriculum on => phase 0 FADE, rest PENDING; curriculum
    off => all OFF (native kickoff + default_reward, no drill ever fires)."""
    start = "FADE" if bool(job.get("curriculum", True)) else "OFF"
    phases = []
    for i, p in enumerate(job["phases"]):
        phases.append(_new_phase(p["name"], p["metric"], p["x"], p["y"], p["z"],
                                 p.get("reward"), p["drill"],
                                 start if i == 0 else ("PENDING" if start == "FADE" else "OFF")))
    return phases


def main():
    job_path = sys.argv[1]
    build_dir = sys.argv[2]
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)

    L = _import_module(build_dir)
    state_path = job["out_path"] + ".state.json"
    resuming = os.path.exists(state_path)

    # Layout: BC arm -> convert the .pt to WEIG and read its recorded spec; random arm -> job's explicit
    # layout. Seeding only on a FRESH run (a resume restores trained weights via Load(), not BC re-seed).
    if job.get("checkpoint"):
        weig = os.path.join(job["save_dir"], "_seed_member_0")
        os.makedirs(job["save_dir"], exist_ok=True)
        convert_checkpoint(L, job["checkpoint"], weig)
        layout = read_spec(weig)
    else:
        weig = None
        layout = {k: job[k] for k in
                  ("entity_features", "query_features", "kv_features", "n_tokens", "n_controls")}
        layout["head"] = job["head"]
        layout["profile"] = True
        layout["n_frames"] = int(job.get("n_frames", 13))

    learner = build_learner(L, job, layout)
    if weig and not resuming:
        learner.seed_member(0, weig)
        print(f"[cpp-ab2] seeded member 0 from {weig}", flush=True)
    learner.start_agents()

    epochs = int(job["epochs"])
    out_path = job["out_path"]
    native_time = float(job["native_time"])
    native_fraction = float(job.get("native_fraction", 0.0))
    # FADE-walk metric smoothing window (epochs) for phases that don't set their own. See advance_phase.
    default_smooth = int(job.get("smooth_window", 0))

    if resuming:
        with open(state_path, "r", encoding="utf-8") as f:
            st = json.load(f)
        if int(st.get("schema_version", 1)) != STATE_SCHEMA_VERSION:
            raise SystemExit(
                f"[cpp-ab2] {state_path} is a pre-drill-rework (v1) state file; the sequential-phase "
                f"curriculum semantics are incompatible. Delete it (and the run's checkpoints if you "
                f"want a clean restart) and re-run.")
        phases = st["phases"]
        done = int(st["done_epochs"])
        curve = list(st["curve"])
        print(f"[cpp-ab2 {job['arm']}/seed{job['seed']}] resumed at epoch {done}/{epochs} "
              f"({' '.join(p['name'] + '=' + p['state'] for p in phases)})", flush=True)
    else:
        phases = fresh_phases(job)
        done = 0
        curve = []

    try:
        for it in range(done, epochs):
            # Push the resolved drill config + reward program onto the live envs BEFORE collecting.
            # Exactly the active phase's drill is enabled (always-drill); all retired => native kickoff.
            learner.set_drill_config(build_drill_specs(L, phases), native_time, native_fraction)
            terms, ts, op = split_reward(phase_reward(phases, job))
            learner.set_member_reward(0, terms, ts, op)

            reports = learner.run_iteration()
            d = reports[0].as_dict() if hasattr(reports[0], "as_dict") else dict(reports[0])
            er = float(d.get("Average Episode Reward", float("nan")))
            entry = {"epoch": it, "ep_reward_mean": er,
                     **{p["name"] + "_state": p["state"] for p in phases}}
            for key in ("goals_for_mean", "touches_for_mean"):
                if key in d:
                    entry[key] = float(d[key])
            curve.append(entry)

            # Advance the ACTIVE phase from this epoch's metric; when it retires, unlock the next.
            act = active_phase(phases)
            if act is not None:
                metric = float(d.get(act["metric"], 0.0))
                sw = int(act.get("smooth_window", default_smooth))
                prev = act["state"]
                idx = phases.index(act)
                phases[idx] = advance_phase(act, metric, sw)
                new = phases[idx]["state"]
                if new != prev:
                    nxt = phases[idx + 1] if idx + 1 < len(phases) else None
                    if new == "OFF" and nxt is not None and nxt["state"] == "PENDING":
                        nxt["state"] = "FADE"
                        print(f"[cpp-ab2 {job['arm']}/seed{job['seed']}] epoch {it}: {act['name']} "
                              f"retired -> {nxt['name']} UNLOCKED (FADE)", flush=True)
                    elif new == "OFF":
                        print(f"[cpp-ab2 {job['arm']}/seed{job['seed']}] epoch {it}: {act['name']} "
                              f"retired -> NATIVE (default_reward)", flush=True)
                    else:
                        print(f"[cpp-ab2 {job['arm']}/seed{job['seed']}] epoch {it}: {act['name']} "
                              f"{prev} -> {new}", flush=True)

            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({"schema_version": STATE_SCHEMA_VERSION, "phases": phases,
                           "done_epochs": it + 1, "curve": curve}, f)
            act = active_phase(phases)
            act_str = (f"{act['name']}={act['state']}({act['score']:.0f}/{act['z']:.0f})"
                       if act else "NATIVE")
            print(f"[cpp-ab2 {job['arm']}/seed{job['seed']}] epoch {it}: er={er:.4f} {act_str}",
                  flush=True)
    finally:
        learner.stop_agents()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"arm": job["arm"], "seed": job["seed"], "curve": curve}, f)
    print(f"[cpp-ab2] wrote {out_path}")


if __name__ == "__main__":
    main()
