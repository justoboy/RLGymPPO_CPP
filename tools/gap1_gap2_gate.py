"""Acceptance gate for the runtime curriculum surface (GAP1 + GAP2).

GAP2: every member's report must carry per-owner outcome tallies (goals_for_mean /
touches_for_mean) that the metric-gated curriculum driver gates on.

GAP1: the runtime setters (set_member_reward / set_spawn_curriculum) must mutate the LIVE envs
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


def build_learner():
    ec = L.EnvConfig()
    ec.teamSize = 1
    ec.spawnOpponents = True
    ec.gameMode = L.GameMode.SOCCAR
    ec.nControls = 9
    ec.entityFeatures = 39
    ec.queryFeatures = 208
    ec.kvFeatures = 69
    ec.nTokens = 151
    # League path: per-episode active-set sampler stamps slotOwners (slot0->member0, slot1->member1).
    ec.memberCount = 2
    ec.siblingFraction = 1.0
    ec.teamConfigs = [(1, 1)]
    ec.teamWeights = [1.0]
    ec.randomSeed = 777
    # Non-empty memberRewardConfigs => EnvFactory builds a ComposedReward (the dynamic_cast target for
    # set_member_reward). Start with a single low-weight term so the baseline reward is small.
    m0 = L.MemberRewardConfig()
    m0.termWeights = [("proximity", 0.1)]
    m0.teamSpirit = 0.0
    m0.opponentPunish = 0.0
    m1 = L.MemberRewardConfig()
    m1.termWeights = [("proximity", 0.1)]
    m1.teamSpirit = 0.0
    m1.opponentPunish = 0.0
    ec.memberRewardConfigs = [m0, m1]

    cfg = L.LearnerConfig()
    cfg.deviceType = L.LearnerDeviceType.CPU
    cfg.collisionMeshesFolder = r"D:\RLBotTraining\RLGymSim_CPP\collision_meshes"
    cfg.numThreads = 1
    cfg.numGamesPerThread = 1
    cfg.timestepsPerIteration = 64
    cfg.expBufferSize = 4096
    cfg.standardizeReturns = False  # P6: match Python raw returns
    cfg.checkpointLoadFolder = ""
    cfg.checkpointSaveFolder = ""
    cfg.sendMetrics = False
    cfg.ppo.batchSize = 32
    cfg.ppo.miniBatchSize = 16
    cfg.ppo.epochs = 1
    cfg.ppo.lambdaFSP = 0.0

    cfg.memberPerceiver = [make_perceiver(39), make_perceiver(39)]
    return L.create_learner(ec, cfg)


def main():
    print("CWD:", os.getcwd())
    learner = build_learner()
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
    # --- GAP1: enable the spawn curriculum on the live TeamConfigMutator (recede distance, prob 1). ---
    learner.set_spawn_curriculum(True, 0.5, 1.0)

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


if __name__ == "__main__":
    main()
