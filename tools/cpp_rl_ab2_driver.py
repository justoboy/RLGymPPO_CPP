"""C++-learner curriculum driver for the rl_ab2 BC-vs-random / discrete-vs-continuous ablation.

Runs under anaconda 3.9 (it imports the cp39 rlgym_learner .pyd). ONE invocation trains ONE (arm, seed)
run: a single-member learner (every slot owned by member 0 = self-play) whose head type + init come from
the job, driven through the same two-stage metric-gated curriculum as the Python train_rl_ab2.py:

    touch  -> gate on touch_streak consecutive iterations whose touches_for_mean >= touch_gate
    goal   -> gate on goal_streak consecutive iterations whose goals_for_mean >= goal_gate
    blending -> linearly fade ratio 1 -> 0 over ramp_epochs (reward scaffold fades + spawn recedes)
    default -> ratio=0, native kickoff, DEFAULT_REWARD

The learner exposes the runtime surface this needs: run_iteration() returns per-member Reports carrying
goals_for_mean / touches_for_mean; set_member_reward / set_spawn_curriculum mutate the live envs in place.
Seeding is optional (random arm = no checkpoint -> stay at the random init create_learner built). One
learner per arm is realized by one invocation per (arm, seed); the Python side runs arms sequentially.

The job JSON (written by the reference implementation's bridge) carries everything: the BC .pt to convert
(or null for the random arm), the bootstrap/default reward dicts to blend per iteration, the gate/streak/
ramp params, and the out_path where the per-run curve + resumable curriculum state are persisted. RESUME:
if <out_path>.state.json exists, the phase / ratio / streak / done / curve are restored and the learner
Load()s its weights from load_dir (no re-seed). Layout comes from the converted spec.json (BC arms) or the
job's explicit layout (random arm).
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


def blend_reward(ratio, bootstrap, default):
    """Per-term linear blend of scaffold and real objective by `ratio` (1 = pure scaffold, 0 = default).
    Reserved keys (team_spirit/opponent_punish) are equal in both sets so they stay constant; we return
    them split out as scalars. Mirrors the reference implementation's blend_reward."""
    out = {}
    for k in set(bootstrap) | set(default):
        bw = float(bootstrap.get(k, 0.0))
        dw = float(default.get(k, 0.0))
        out[k] = ratio * bw + (1.0 - ratio) * dw
    ts = out.pop("team_spirit", 0.6)
    op = out.pop("opponent_punish", 1.0)
    return out, ts, op


def build_learner(L, job, layout):
    """Single-member learner (every slot owned by member 0 = self-play). head from job.head / layout."""
    init_terms, init_ts, init_op = blend_reward(
        float(job["ratio_init"]), job["bootstrap_reward"], job["default_reward"])
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
        spawn_curriculum_enabled=False, spawn_curriculum_ratio=1.0,
        spawn_curriculum_prob=float(job["spawn_prob"]))

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
    if resuming:
        with open(state_path, "r", encoding="utf-8") as f:
            st = json.load(f)
        phase = st["phase"]
        ratio = float(st["ratio"])
        streak = int(st["streak"])
        done = int(st["done_epochs"])
        curve = list(st["curve"])
        print(f"[cpp-ab2 {job['arm']}/seed{job['seed']}] resumed at epoch {done}/{epochs} "
              f"(phase={phase} ratio={ratio:.2f})", flush=True)
    else:
        phase = "touch" if job["curriculum"] else "default"
        ratio = 1.0 if job["curriculum"] else 0.0
        streak = 0
        done = 0
        curve = []

    try:
        for it in range(done, epochs):
            terms, ts, op = blend_reward(ratio, job["bootstrap_reward"], job["default_reward"])
            learner.set_member_reward(0, terms, ts, op)
            learner.set_spawn_curriculum(phase != "default", ratio, float(job["spawn_prob"]))
            reports = learner.run_iteration()
            d = reports[0].as_dict() if hasattr(reports[0], "as_dict") else dict(reports[0])
            goals_mean = float(d.get("goals_for_mean", 0.0))
            touch_mean = float(d.get("touches_for_mean", 0.0))
            er = float(d.get("Average Episode Reward", float("nan")))
            curve.append({"epoch": it, "ep_reward_mean": er, "goals_for_mean": goals_mean,
                          "touches_for_mean": touch_mean, "phase": phase, "ratio": ratio})
            # advance the two-stage curriculum for the NEXT iteration
            if phase == "touch":
                streak = streak + 1 if touch_mean >= job["touch_gate"] else 0
                if streak >= job["touch_streak"]:
                    phase, streak = "goal", 0
            elif phase == "goal":
                streak = streak + 1 if goals_mean >= job["goal_gate"] else 0
                if streak >= job["goal_streak"]:
                    phase, streak = "blending", 0
                    ratio = max(0.0, 1.0 - 1.0 / job["ramp_epochs"])
            elif phase == "blending":
                ratio = max(0.0, ratio - 1.0 / job["ramp_epochs"])
                if ratio <= 1e-9:
                    ratio, phase = 0.0, "default"
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({"phase": phase, "ratio": ratio, "streak": streak,
                           "done_epochs": it + 1, "curve": curve}, f)
            print(f"[cpp-ab2 {job['arm']}/seed{job['seed']}] epoch {it}: er={er:.4f} "
                  f"goals={goals_mean:.2f} touches={touch_mean:.2f} phase={phase} ratio={ratio:.2f}",
                  flush=True)
    finally:
        learner.stop_agents()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"arm": job["arm"], "seed": job["seed"], "curve": curve}, f)
    print(f"[cpp-ab2] wrote {out_path}")


if __name__ == "__main__":
    main()
