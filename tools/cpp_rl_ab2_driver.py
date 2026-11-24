"""C++-learner curriculum driver for the rl_ab2 BC-vs-random / discrete-vs-continuous ablation.

Runs under anaconda 3.9 (it imports the cp39 rlgym_learner .pyd). ONE invocation trains ONE (arm, seed)
run: a single-member learner (every slot owned by member 0 = self-play) whose head type + init come from
the job, driven through the SAME two-wheel (touch + goal) spawn curriculum as the Python reference
train_rl_ab2.py. Each wheel is an INDEPENDENT state machine the driver advances per EPOCH:

    ACQUIRING: streak = streak+1 if metric >= x else 0;  streak >= y  -> FADE, score = 0
    FADE:      score = clamp(score + (1 if metric >= x else -2), 0, z); score >= z -> wheel OFF
    g = score / z   (0 in ACQUIRING, climbs 0->1 through FADE)

    touch drill geometry: car at d = 400 + 3600*g_touch from a center ball (g=0 -> ball on car)
    goal  drill geometry: ball at +5120*(1-g_goal) on the lone BLUE car's +y attack axis (g=0 -> easy goal)
    per-wheel fire prob:  p = prob_floor + (1-prob_floor)*(1-g)   (g=0 -> p=1 always drill; g=1 -> floor)

The C++ learner auto-resets each game internally, so the per-episode drill Bernoulli lives in the mutator;
the driver only advances each wheel per EPOCH and pushes the resulting geometry + probability + clock caps
via set_wheel_config(). The metric feeding each wheel is the plain ALL-episode mixed mean (touches_for_mean
for touch, goals_for_mean for goal) — no drill-only gating; the probability floor bounds the dilution.

Reward is a STEP, not a blend: while either wheel is active the env uses the fixed boosted phase-1 set; once
BOTH wheels are OFF it steps to the DEFAULT set (win_prob arrives). No linear reward blend.

The job JSON (written by the reference implementation's bridge) carries everything: the BC .pt to convert
(or null for the random arm), the phase-1 + default reward maps, the per-wheel x/y/z + prob_floor + drill_time
/ native_time, and out_path where the per-run curve + resumable wheel state is persisted. RESUME: if
<out_path>.state.json exists, both wheel states + done_epochs + curve are restored and the learner Load()s
its weights from load_dir (no re-seed). Layout comes from the converted spec.json (BC arms) or the job's
explicit layout (random arm).
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


def split_reward(rmap):
    """Split a reward map into (term_weights_without_reserved, team_spirit, opponent_punish). The reserved
    keys team_spirit/opponent_punish are passed to set_member_reward as scalars, mirroring the reference."""
    m = dict(rmap)
    ts = float(m.pop("team_spirit", 0.6))
    op = float(m.pop("opponent_punish", 1.0))
    return m, ts, op


def advance_wheel(w, metric):
    """Advance one wheel's state machine by one epoch given its mixed-mean metric. Returns the new wheel dict
    {state, streak, score}. ACQUIRING -> FADE on streak>=y; FADE -> OFF on score>=z."""
    state = w["state"]
    if state == "OFF":
        return w
    hit = metric >= w["x"]
    if state == "ACQUIRING":
        streak = w["streak"] + 1 if hit else 0
        if streak >= w["y"]:
            return {"state": "FADE", "streak": 0, "score": 0.0,
                    "x": w["x"], "y": w["y"], "z": w["z"]}
        return {"state": "ACQUIRING", "streak": streak, "score": 0.0,
                "x": w["x"], "y": w["y"], "z": w["z"]}
    # FADE
    score = max(0.0, min(float(w["z"]), w["score"] + (1.0 if hit else -2.0)))
    if score >= w["z"]:
        return {"state": "OFF", "streak": 0, "score": float(w["z"]),
                "x": w["x"], "y": w["y"], "z": w["z"]}
    return {"state": "FADE", "streak": 0, "score": score,
            "x": w["x"], "y": w["y"], "z": w["z"]}


def wheel_geometry(w):
    """(enabled, g) for a wheel: enabled = not OFF; g = score/z (0 in ACQUIRING, climbs through FADE)."""
    enabled = w["state"] != "OFF"
    g = (w["score"] / w["z"]) if w["z"] > 0 else 0.0
    return enabled, g


def fire_prob(g, prob_floor):
    """Per-wheel fire probability: floor at g=1, 1.0 at g=0 (always drill while easy)."""
    return prob_floor + (1.0 - prob_floor) * (1.0 - g)


def build_learner(L, job, layout):
    """Single-member learner (every slot owned by member 0 = self-play). head from job.head / layout."""
    init_terms, init_ts, init_op = split_reward(job["phase1_reward"])
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
        max_overtime_seconds=float(job["max_overtime_seconds"]))

    pc = build_perceiver(L, layout)
    cfg = L.LearnerConfig()
    cfg.deviceType = getattr(L.LearnerDeviceType, str(job["device"]))
    cfg.collisionMeshesFolder = os.environ.get(
        "COLLISION_MESHES", r"D:\RLBotTraining\RLGymSim_CPP\collision_meshes")
    cfg.numThreads = int(job["num_threads"])
    cfg.numGamesPerThread = int(job["games_per_thread"])
    cfg.timestepsPerIteration = int(job["timesteps_per_iteration"])
    cfg.expBufferSize = int(job["exp_buffer_size"])
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

    if resuming:
        with open(state_path, "r", encoding="utf-8") as f:
            st = json.load(f)
        touch = st["touch"]
        goal = st["goal"]
        done = int(st["done_epochs"])
        curve = list(st["curve"])
        print(f"[cpp-ab2 {job['arm']}/seed{job['seed']}] resumed at epoch {done}/{epochs} "
              f"(touch={touch['state']} goal={goal['state']})", flush=True)
    else:
        # curriculum off => both wheels start OFF (native kickoff + DEFAULT_REWARD, no drill ever fires);
        # curriculum on => both wheels start ACQUIRING (the two-wheel curriculum is active from epoch 0).
        init_state = "ACQUIRING" if bool(job.get("curriculum", True)) else "OFF"
        touch = {"state": init_state, "streak": 0, "score": 0.0,
                 "x": float(job["x_touch"]), "y": int(job["y_touch"]), "z": float(job["z_touch"])}
        goal = {"state": init_state, "streak": 0, "score": 0.0,
                "x": float(job["x_goal"]), "y": int(job["y_goal"]), "z": float(job["z_goal"])}
        done = 0
        curve = []

    try:
        for it in range(done, epochs):
            # Push the current wheel geometry + fire-probability onto the live envs BEFORE collecting.
            t_en, g_t = wheel_geometry(touch)
            g_en, g_g = wheel_geometry(goal)
            learner.set_wheel_config(
                t_en, TOUCH_MIN_DIST + (TOUCH_MAX_DIST - TOUCH_MIN_DIST) * g_t, fire_prob(g_t, prob_floor),
                g_en, GOAL_MAX_Y * (1.0 - g_g), fire_prob(g_g, prob_floor),
                drill_time, native_time)
            # Reward steps (no blend): boosted phase-1 while either wheel active, default once both OFF.
            rmap = job["default_reward"] if (touch["state"] == "OFF" and goal["state"] == "OFF") \
                else job["phase1_reward"]
            terms, ts, op = split_reward(rmap)
            learner.set_member_reward(0, terms, ts, op)

            reports = learner.run_iteration()
            d = reports[0].as_dict() if hasattr(reports[0], "as_dict") else dict(reports[0])
            goals_mean = float(d.get("goals_for_mean", 0.0))
            touch_mean = float(d.get("touches_for_mean", 0.0))
            er = float(d.get("Average Episode Reward", float("nan")))
            curve.append({"epoch": it, "ep_reward_mean": er, "goals_for_mean": goals_mean,
                          "touches_for_mean": touch_mean,
                          "touch_state": touch["state"], "goal_state": goal["state"]})

            # Advance each wheel's state machine from this epoch's mixed means (for the NEXT epoch).
            touch = advance_wheel(touch, touch_mean)
            goal = advance_wheel(goal, goals_mean)

            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({"touch": touch, "goal": goal, "done_epochs": it + 1, "curve": curve}, f)
            print(f"[cpp-ab2 {job['arm']}/seed{job['seed']}] epoch {it}: er={er:.4f} "
                  f"goals={goals_mean:.2f} touches={touch_mean:.2f} "
                  f"touch={touch['state']} goal={goal['state']}", flush=True)
    finally:
        learner.stop_agents()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"arm": job["arm"], "seed": job["seed"], "curve": curve}, f)
    print(f"[cpp-ab2] wrote {out_path}")


if __name__ == "__main__":
    main()
