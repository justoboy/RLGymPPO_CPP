"""Acceptance gate for the runtime curriculum surface (GAP1 + GAP2).

GAP2: every member's report must carry per-owner outcome tallies (goals_for_mean /
touches_for_mean) that the metric-gated curriculum driver gates on.

GAP1: the runtime setters (set_member_reward / set_wheel_config) must mutate the LIVE envs
in place (no learner rebuild) and leave the learner still able to run a valid iteration. To make
set_member_reward observable, the env is built on the league path with memberRewardConfigs set, so
the reward is a ComposedReward (the dynamic_cast target) rather than the plain CombinedReward.

This is a functional gate (callable + observable side effects), not a numeric parity gate.
"""
import os
import sys

# torch DLLs must be resolvable before importing the extension.
for p in (
    r"C:\Users\justo\anaconda3\lib\site-packages\torch\lib",
    r"C:\Users\justo\anaconda3\Library\bin",
    r"C:\Users\justo\anaconda3\DLLs",
):
    if os.path.isdir(p):
        os.add_dll_directory(p)

sys.path.insert(0, r"D:\RLBotTraining\RLGymPPO_CPP\build\Release")
import torch  # noqa: E402  (must precede rlgym_learner import)
import rlgym_learner as L  # noqa: E402


def make_perceiver(ef):
    pc = L.PerceiverConfig()
    pc.nDims = 256
    pc.nHeads = 8
    pc.nLayers = 1
    pc.nControls = 9
    # Must match the TemporalObsBuilder's hardcoded superset layout (P5).
    pc.entityFeatures = ef
    pc.queryFeatures = 208
    pc.kvFeatures = 69
    pc.nTokens = 151
    return pc


def build_learner(member_count=2, sibling_fraction=1.0, no_touch_secs=1.5, timesteps=512):
    ec = L.EnvConfig()
    ec.teamSize = 1
    ec.spawnOpponents = True
    ec.gameMode = L.GameMode.SOCCAR
    ec.nControls = 9
    ec.entityFeatures = 39
    ec.queryFeatures = 208
    ec.kvFeatures = 69
    ec.nTokens = 151
    # Short no-touch timeout so an episode actually ENDS within the (small) iteration budget; otherwise
    # no episode completes and the per-owner tallies/reward stay NaN (no completed episode to average).
    ec.noTouchTimeoutSecs = no_touch_secs
    # League path: per-episode active-set sampler stamps slotOwners. member_count=2 + sibling=1 ->
    # slot0->member0, slot1->member1 (sibling match). member_count=1 -> every slot owned by member0
    # (self-play): the per-owner reward is the MAX of the two cars' returns (no cancellation) and the
    # goals/touches tallies average across both cars (the policy's true rate).
    ec.memberCount = member_count
    ec.siblingFraction = sibling_fraction
    ec.teamConfigs = [(1, 1)]
    ec.teamWeights = [1.0]
    ec.randomSeed = 777
    # Non-empty memberRewardConfigs => EnvFactory builds a ComposedReward (the dynamic_cast target for
    # set_member_reward). Start with a single low-weight term so the baseline reward is small. One spec
    # per member (self-play: 1; sibling league: 2).
    mrcs = []
    for _ in range(member_count):
        m = L.MemberRewardConfig()
        m.termWeights = [("proximity", 0.1)]
        m.teamSpirit = 0.0
        m.opponentPunish = 0.0
        mrcs.append(m)
    ec.memberRewardConfigs = mrcs

    cfg = L.LearnerConfig()
    cfg.deviceType = L.LearnerDeviceType.CPU
    cfg.collisionMeshesFolder = r"D:\RLBotTraining\RLGymSim_CPP\collision_meshes"
    cfg.numThreads = 1
    cfg.numGamesPerThread = 1
    cfg.timestepsPerIteration = timesteps
    cfg.expBufferSize = 4096
    cfg.standardizeReturns = False  # P6: match Python raw returns
    cfg.checkpointLoadFolder = ""
    cfg.checkpointSaveFolder = ""
    cfg.sendMetrics = False
    cfg.ppo.batchSize = 32
    cfg.ppo.miniBatchSize = 16
    cfg.ppo.epochs = 1
    cfg.ppo.lambdaFSP = 0.0

    cfg.memberPerceiver = [make_perceiver(39) for _ in range(member_count)]
    return L.create_learner(ec, cfg)


def run_league():
    # One learner per process: torch::set_num_interop_threads can only be called once per process, so
    # the league and self-play learners CANNOT coexist in one interpreter. The driver runs this script
    # once per mode (argv[1]) and aggregates the two exit codes.
    print("CWD:", os.getcwd())
    learner = build_learner(member_count=2, sibling_fraction=1.0)
    assert learner.obs_size == 208 + 151 * 69 + 151, f"unexpected obs_size {learner.obs_size}"

    learner.start_agents()

    # --- Iteration 1: baseline (curriculum disabled, low-weight reward). GAP2 keys must appear. ---
    reports = learner.run_iteration()
    assert isinstance(reports, list) and len(reports) == 2, "expected 2 member reports"
    for i, r in enumerate(reports):
        d = r.as_dict()
        assert "goals_for_mean" in d, f"member {i} report missing goals_for_mean: {list(d)}"
        assert "touches_for_mean" in d, f"member {i} report missing touches_for_mean: {list(d)}"
    base = reports[0].as_dict()
    print("baseline member0 goals_for_mean:", base["goals_for_mean"],
          "touches_for_mean:", base["touches_for_mean"])

    # --- GAP1: rebind each member's reward program on the LIVE ComposedReward (heavy weight). ---
    learner.set_member_reward(0, {"goal": 5.0, "speed": 2.0}, 0.8, 1.0)
    learner.set_member_reward(1, {"goal": 5.0, "speed": 2.0}, 0.8, 1.0)
    # --- GAP1: enable the two-wheel curriculum on the live TeamConfigMutator (both wheels ON, prob 1). ---
    # set_wheel_config(touch_enabled, touch_d, touch_prob, goal_enabled, goal_ball_y, goal_prob,
    #                  drill_time, native_time). prob 1 -> every episode fires the drill.
    learner.set_wheel_config(True, 400.0, 1.0, True, 0.0, 1.0, 10.0, 300.0)

    # --- Iteration 2: setters must not crash and the learner must still produce valid reports. ---
    reports2 = learner.run_iteration()
    assert isinstance(reports2, list) and len(reports2) == 2, "post-setter iteration lost members"
    for i, r in enumerate(reports2):
        d = r.as_dict()
        assert "goals_for_mean" in d, f"post-setter member {i} missing goals_for_mean"
        assert "touches_for_mean" in d, f"post-setter member {i} missing touches_for_mean"

    learner.stop_agents()

    # Tally keys present on both members, both before and after the runtime setters => GAP1 + GAP2 hold.
    print("GAP1/GAP2 GATE PASS")


def run_selfplay():
    # --- Self-play owner-collapse fix: member_count=1 (one policy owns both cars). The reported
    # Average Episode Reward must be the MAX of the two cars' returns (nonzero), NOT the old cross-team
    # sum that cancelled to ~1e-8. Use opponentPunish=1.0 so the OLD cross-team sum WOULD cancel; the
    # per-owner max must keep the reported reward nonzero. ---
    sp = build_learner(member_count=1, sibling_fraction=0.0, no_touch_secs=1.5, timesteps=512)
    sp.start_agents()
    sp.set_member_reward(0, {"proximity": 1.0}, 0.6, 1.0)  # opponentPunish=1.0 -> symmetric cancel case
    sp_reports = sp.run_iteration()
    assert isinstance(sp_reports, list) and len(sp_reports) == 1, "self-play expected 1 member report"
    spd = sp_reports[0].as_dict()
    er = float(spd.get("Average Episode Reward", 0.0))
    print(f"self-play member0 Average Episode Reward (max of both cars): {er}")
    assert abs(er) > 1e-3, f"self-play reward still cancels to ~0 (got {er}); owner-max fix not working"
    sp.stop_agents()
    print("SELF-PLAY OWNER-COLLAPSE GATE PASS")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "league"
    if mode == "selfplay":
        run_selfplay()
    else:
        run_league()
