"""C++-learner curriculum driver for the rl_ab2 BC-vs-random / discrete-vs-continuous ablation.

Runs under anaconda 3.9 (it imports the cp39 rlgym_learner .pyd). ONE invocation trains ONE (arm, seed)
run: a single-member learner (every slot owned by member 0 = self-play) whose head type + init come from
the job, driven through the SAME THREE-WHEEL (touch + goal + duel) spawn curriculum as the reference
implementation's train_rl_ab2.py. Each wheel is an INDEPENDENT state machine the driver advances per EPOCH
(FADE-FIRST-THEN-GATE -- never ask for native competence before the bot has been faded INTO native):

    FADE: score = clamp(score + (metric >= x ? +2 : -3), 0, z); score >= z -> GATE (streak = 0)
          The penalty (-3) EXCEEDS the reward (+2): the walk climbs only above the ~60% break-even, so a
          lucky epoch cannot carry a wheel forward (this kills the old +1/-2 limit cycle).
    GATE: streak = streak+1 if metric >= x else 0; streak >= y -> OFF
          The wheel's own drill is FULLY SUPPRESSED during GATE, so the metric is measured on the native
          fallback episode; y consecutive native epochs meeting x proves real competence before retirement.
    g = score / z is the wheel's fade fraction (drill probability p = floor + (1-floor)*(1-g)).

    touch drill geometry: 1v0 -- ball at center, lone car at d = 400 + 3600*g_touch facing it.
    goal  drill geometry: 1v0 breakaway -- lone car behind a ball on its +y attack axis at
                          |y| = 5120*(1-g_goal) (opponent absent = uncontested finish).
    GATE fallback:        the hardest 1v0 -- ball at center, lone car parked at kickoff distance on its
                          attack axis, drill-active so a goal respawns (the episode the GATE measures on).
    duel  wheel (phase 2): LOCKED until touch+goal retire, then native 1v1 kickoff under DUEL_REWARD with
                          the clock capped at kickoff_time (hard-finish, no OT); metric = goals_for_mean
                          (a goal requires a touch, so goals subsumes touches).

THREE-PHASE REWARD (a STEP, never a blend): PHASE 1 (touch or goal wheel not OFF) -> PHASE1_REWARD;
PHASE 2 (both retired, duel not OFF) -> DUEL_REWARD; PHASE 3 (all OFF) -> DEFAULT_REWARD. The C++ mutator
forces the 1v0 lineup + GATE fallback whenever phase1_active, so the driver only writes per-wheel enabled
(= FADING), geometry, fire-probability, the phase1_active flag, and the duel clock cap via set_wheel_config.
GOAL-FIRST PRECEDENCE (both drills would fire -> the 1v0 goal drill wins) lives in the mutator.

The job JSON (written by the reference implementation's bridge) carries everything: the BC .pt to convert
(or null for the random arm), the phase-1 + duel + default reward maps, the per-wheel x/y/z + prob_floor +
drill_time / native_time (+ optional kickoff_time, default 30s), and out_path where the per-run curve +
resumable wheel state is persisted. RESUME: if <out_path>.state.json exists, all three wheel states +
done_epochs + curve are restored (with migration: a missing duel wheel is added LOCKED, or FADE when
touch+goal already retired; any legacy ACQUIRING becomes FADE) and the learner Load()s its weights from
load_dir (no re-seed). Layout comes from the converted spec.json (BC arms) or the job's explicit layout.
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

# Touch-drill car->ball distance range (matches the C++ mutator's SPAWN_MIN/MAX_DIST).
TOUCH_MIN_DIST = 400.0
TOUCH_MAX_DIST = 4000.0
# Goal-drill ball |y| at full difficulty (g=0 -> ball at the goal mouth = easiest, g=1 -> center = hardest).
GOAL_MAX_Y = 5120.0
# Phase-2 duel clock cap (s) when the job omits kickoff_time (the reference's default).
DEFAULT_KICKOFF_TIME = 30.0


def split_reward(rmap):
    """Split a reward map into (term_weights_without_reserved, team_spirit, opponent_punish). The reserved
    keys team_spirit/opponent_punish are passed to set_member_reward as scalars, mirroring the reference."""
    m = dict(rmap)
    ts = float(m.pop("team_spirit", 0.6))
    op = float(m.pop("opponent_punish", 1.0))
    return m, ts, op


def advance_wheel(w, metric, smooth_window=0):
    """Advance one wheel's state machine by one epoch given its epoch-mean metric (FADE-FIRST-THEN-GATE).
    FADE: score = clamp(score + (+2 if hit else -3), 0, z); score >= z -> GATE (streak = 0).
    GATE: streak = streak+1 if RAW metric >= x else 0; streak >= y -> OFF. No-op once OFF or LOCKED.

    SMOOTHING (smooth_window > 0): the FADE walk's `hit` is decided on the ROLLING MEAN of the last
    `smooth_window` epoch-metrics, not the single raw epoch. This collapses the per-epoch sampling
    variance that otherwise turns the +2/-3 walk into a diffusion-dominated random walk absorbed at the
    0 floor (a true mean just above x still dips below it ~half the time on 32-match noise, so the tiny
    drift never accumulates). Smoothing concentrates the estimator on the true mean: a wheel genuinely
    above x climbs reliably, one genuinely below stays pinned -- it SHARPENS the ~60% bar rather than
    lowering it, so luck still cannot carry a wheel forward. The GATE proof stays on the RAW metric (a
    clean native-competence test, never smoothed). The rolling buffer lives in w["hist"] (persisted)."""
    state = w["state"]
    if state in ("OFF", "LOCKED"):
        return w
    hist = list(w.get("hist", []))
    if smooth_window > 0:
        hist = (hist + [float(metric)])[-smooth_window:]
        hit = (sum(hist) / len(hist)) >= w["x"] if hist else (metric >= w["x"])
    else:
        hit = metric >= w["x"]
    if state == "FADE":
        score = max(0.0, min(float(w["z"]), w["score"] + (2.0 if hit else -3.0)))
        if score >= w["z"]:
            return {"state": "GATE", "streak": 0, "score": float(w["z"]),
                    "x": w["x"], "y": w["y"], "z": w["z"], "hist": []}
        return {"state": "FADE", "streak": 0, "score": score,
                "x": w["x"], "y": w["y"], "z": w["z"], "hist": hist}
    # GATE: the wheel's drill is suppressed (the mutator measures on the native fallback episode).
    # The retirement proof is the RAW metric -- never smoothed -- so a wheel only retires on genuine
    # native competence, not on a smoothed average of easier fading epochs.
    raw_hit = metric >= w["x"]
    streak = w["streak"] + 1 if raw_hit else 0
    if streak >= w["y"]:
        return {"state": "OFF", "streak": 0, "score": w["score"],
                "x": w["x"], "y": w["y"], "z": w["z"], "hist": []}
    return {"state": "GATE", "streak": streak, "score": w["score"],
            "x": w["x"], "y": w["y"], "z": w["z"], "hist": hist}


def wheel_geometry(w):
    """(firing, g) for a wheel: firing = the drill may spawn (FADE only; GATE/OFF suppress it);
    g = score/z is the fade fraction (geometry eases from easy at g=0 to hard at g=1)."""
    firing = w["state"] == "FADE"
    g = (w["score"] / w["z"]) if w["z"] > 0 else 0.0
    return firing, g


def fire_prob(g, prob_floor):
    """Per-wheel fire probability: floor at g=1, 1.0 at g=0 (always drill while easy)."""
    return prob_floor + (1.0 - prob_floor) * (1.0 - g)


def phase_reward(wheels, job):
    """Three-phase reward program (a STEP, no blending): PHASE1 while touch or goal is not OFF; DUEL once
    both retire while the duel wheel is not OFF; DEFAULT when all three are OFF. A job without duel_reward
    (older bridge) falls back to default_reward so phase 2 still runs, just unscaffolded."""
    if wheels["touch"]["state"] != "OFF" or wheels["goal"]["state"] != "OFF":
        return dict(job["phase1_reward"])
    if wheels["duel"]["state"] != "OFF":
        return dict(job.get("duel_reward") or job["default_reward"])
    return dict(job["default_reward"])


def build_learner(L, job, layout):
    """Single-member learner (every slot owned by member 0 = self-play). head from job.head / layout."""
    init_rmap = job["phase1_reward"] if bool(job.get("curriculum", True)) else job["default_reward"]
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
        wheel_enabled=True, wheel_drill_time=float(job["drill_time"]),
        wheel_native_time=float(job["native_time"]),
        max_overtime_seconds=float(job["max_overtime_seconds"]),
        # Native matches run the full native_time + OT (kickoff respawn on goal, no no-touch truncation),
        # matching the reference's rl_ab2 behavior (10s drills, native matches to the clock). This is also
        # what lets the profile tracker accumulate multi-goal history. Drill episodes still cap at
        # drill_time via the drillActive path; the phase-2 duel caps at duel_time via hardFinish.
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


def _new_wheel(state, x, y, z):
    return {"state": state, "streak": 0, "score": 0.0, "x": float(x), "y": int(y), "z": float(z),
            "hist": []}


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
    prob_floor = float(job["prob_floor"])
    drill_time = float(job["drill_time"])
    native_time = float(job["native_time"])
    # The bridge may omit kickoff_time (older jobs); default to the reference's 30s duel cap.
    kickoff_time = float(job.get("kickoff_time", DEFAULT_KICKOFF_TIME))
    # FADE-walk metric smoothing window (epochs); 0 disables smoothing (raw per-epoch hit). See advance_wheel.
    smooth_window = int(job.get("smooth_window", 0))

    if resuming:
        with open(state_path, "r", encoding="utf-8") as f:
            st = json.load(f)
        touch = st["touch"]
        goal = st["goal"]
        done = int(st["done_epochs"])
        curve = list(st["curve"])
        # Migration from pre-duel checkpoints: the phase-2 duel wheel didn't exist yet. If touch+goal
        # already retired (the run had reached native play), start the duel FADE so it re-scaffolds;
        # otherwise LOCKED (it unlocks naturally when touch+goal retire).
        if "duel" not in st:
            retired = touch["state"] == "OFF" and goal["state"] == "OFF"
            duel = _new_wheel("FADE" if retired else "LOCKED",
                              job.get("x_duel", 1.0), job.get("y_duel", 5), job.get("z_duel", 5))
        else:
            duel = st["duel"]
        # Migration from the old ACQUIRING->FADE->OFF machine: ACQUIRING no longer exists (wheels now
        # start in FADE). Any persisted ACQUIRING becomes FADE so the run continues under the new walk.
        for w in (touch, goal, duel):
            if w["state"] == "ACQUIRING":
                w["state"] = "FADE"
            w.setdefault("hist", [])  # migration: pre-smoothing state files have no rolling buffer
        print(f"[cpp-ab2 {job['arm']}/seed{job['seed']}] resumed at epoch {done}/{epochs} "
              f"(touch={touch['state']} goal={goal['state']} duel={duel['state']})", flush=True)
    else:
        # curriculum off => all three wheels OFF (native kickoff + DEFAULT_REWARD, no drill ever fires);
        # curriculum on => touch+goal start FADE (PHASE1_REWARD + easy drill), duel LOCKED (it only
        # activates once touch+goal retire).
        if bool(job.get("curriculum", True)):
            touch = _new_wheel("FADE", job["x_touch"], job["y_touch"], job["z_touch"])
            goal = _new_wheel("FADE", job["x_goal"], job["y_goal"], job["z_goal"])
            duel = _new_wheel("LOCKED", job.get("x_duel", 1.0),
                              job.get("y_duel", 5), job.get("z_duel", 5))
        else:
            touch = _new_wheel("OFF", job["x_touch"], job["y_touch"], job["z_touch"])
            goal = _new_wheel("OFF", job["x_goal"], job["y_goal"], job["z_goal"])
            duel = _new_wheel("OFF", job.get("x_duel", 1.0),
                              job.get("y_duel", 5), job.get("z_duel", 5))
        done = 0
        curve = []

    try:
        for it in range(done, epochs):
            wheels = {"touch": touch, "goal": goal, "duel": duel}
            # Push the current phase + wheel geometry + fire-probability onto the live envs BEFORE
            # collecting. phase1 (touch or goal not OFF) => 1v0 curriculum: each FADING wheel's drill may
            # fire (GATE/OFF suppressed), and when neither fires the mutator uses the 1v0 GATE fallback.
            # phase2 => native 1v1 kickoff capped at kickoff_time (hard-finish, no OT). phase3 => native
            # full match. The mutator rolls the per-episode Bernoulli + applies goal-first precedence.
            phase1 = touch["state"] != "OFF" or goal["state"] != "OFF"
            t_fire, g_t = wheel_geometry(touch)
            g_fire, g_g = wheel_geometry(goal)
            duel_active = duel["state"] != "OFF"
            learner.set_wheel_config(
                t_fire, TOUCH_MIN_DIST + (TOUCH_MAX_DIST - TOUCH_MIN_DIST) * g_t,
                fire_prob(g_t, prob_floor),
                g_fire, GOAL_MAX_Y * (1.0 - g_g), fire_prob(g_g, prob_floor),
                drill_time, native_time,
                phase1, (kickoff_time if (duel_active and not phase1) else -1.0))
            # Reward steps (no blend): PHASE1 boosted dense set, PHASE2 duel scaffold, PHASE3 default.
            terms, ts, op = split_reward(phase_reward(wheels, job))
            learner.set_member_reward(0, terms, ts, op)

            reports = learner.run_iteration()
            d = reports[0].as_dict() if hasattr(reports[0], "as_dict") else dict(reports[0])
            goals_mean = float(d.get("goals_for_mean", 0.0))
            touch_mean = float(d.get("touches_for_mean", 0.0))
            er = float(d.get("Average Episode Reward", float("nan")))
            curve.append({"epoch": it, "ep_reward_mean": er, "goals_for_mean": goals_mean,
                          "touches_for_mean": touch_mean,
                          "touch_state": touch["state"], "goal_state": goal["state"],
                          "duel_state": duel["state"]})

            # Advance each wheel's state machine from this epoch's mixed means (for the NEXT epoch).
            # touch+goal are independent (FADE->GATE->OFF); the duel wheel is LOCKED until BOTH retire,
            # then walks FADE->GATE->OFF on goals_for_mean (a goal requires a touch, so goals subsumes
            # touches and forces the bot to CONVERT the kickoff, not just boot the ball once).
            touch = advance_wheel(touch, touch_mean, smooth_window)
            goal = advance_wheel(goal, goals_mean, smooth_window)
            if duel["state"] == "LOCKED" and touch["state"] == "OFF" and goal["state"] == "OFF":
                duel["state"] = "FADE"
                print(f"[cpp-ab2 {job['arm']}/seed{job['seed']}] epoch {it}: duel UNLOCKED -> "
                      f"PHASE 2 (DUEL_REWARD + 1v1 kickoff, {kickoff_time:.0f}s cap)", flush=True)
            elif duel["state"] not in ("LOCKED", "OFF"):
                prev = duel["state"]
                duel = advance_wheel(duel, goals_mean, smooth_window)
                if duel["state"] != prev:
                    print(f"[cpp-ab2 {job['arm']}/seed{job['seed']}] epoch {it}: duel "
                          f"{prev} -> {duel['state']}"
                          + (" -> PHASE 3 (DEFAULT_REWARD + native kickoff)"
                             if duel["state"] == "OFF" else ""), flush=True)

            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({"touch": touch, "goal": goal, "duel": duel,
                           "done_epochs": it + 1, "curve": curve}, f)
            print(f"[cpp-ab2 {job['arm']}/seed{job['seed']}] epoch {it}: er={er:.4f} "
                  f"goals={goals_mean:.2f} touches={touch_mean:.2f} "
                  f"touch={touch['state']}({touch['score']:.0f}/{touch['z']:.0f}) "
                  f"goal={goal['state']}({goal['score']:.0f}/{goal['z']:.0f}) "
                  f"duel={duel['state']}({duel['score']:.0f}/{duel['z']:.0f})", flush=True)
    finally:
        learner.stop_agents()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"arm": job["arm"], "seed": job["seed"], "curve": curve}, f)
    print(f"[cpp-ab2] wrote {out_path}")


if __name__ == "__main__":
    main()
