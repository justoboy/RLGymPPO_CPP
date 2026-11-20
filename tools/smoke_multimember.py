"""End-to-end smoke test for the multi-member pybind surface (Step 4 gate).

Builds a 2-member learner (member 0 profile-on 39-wide, member 1 profile-off 30-wide), runs one
tiny collect+learn iteration, and asserts run_iteration() returns a list with one dict per member.
This is a functional smoke test, not a parity gate (parity is gated separately by ParityGate).
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
    # The TemporalObsBuilder is hardcoded to the canonical superset layout (P5): query=208,
    # kv=69, tokens=151, entity=39 (profile-on). The perceiver MUST match the obs builder's emitted
    # widths exactly, or the policy input layer is sized wrong and reads out of bounds. Both members
    # share the superset width; the multi-member plumbing is exercised by 2 policies + 2 buffers +
    # per-episode slotOwners routing, not by width divergence.
    pc.entityFeatures = ef  # 39 (profile-on) for both members
    pc.queryFeatures = 208
    pc.kvFeatures = 69
    pc.nTokens = 151
    return pc


def main():
    print("CWD:", os.getcwd())
    print("collision_meshes exists:", os.path.isdir(os.path.join(os.getcwd(), "collision_meshes")))
    ec = L.EnvConfig()
    ec.teamSize = 1
    ec.spawnOpponents = True
    ec.gameMode = L.GameMode.SOCCAR
    ec.nControls = 9
    ec.entityFeatures = 39  # superset profile-on builder width
    # EnvFactory::CreateLearner copies these onto the perceiver (overriding make_perceiver's widths),
    # so they MUST match the TemporalObsBuilder's hardcoded superset layout.
    ec.queryFeatures = 208
    ec.kvFeatures = 69
    ec.nTokens = 151
    # Per-episode active-set routing: 2 members, every episode a sibling 1v1 (slot 0 -> member 0,
    # slot 1 -> member 1). The env's TeamConfigMutator stamps GameState.slotOwners per reset; the
    # learner routes each slot's trajectory to its owner. (Replaces the old frozen cfg.slotOwners.)
    ec.memberCount = 2
    ec.siblingFraction = 1.0  # every episode is a sibling 1v1: slot0->member0, slot1->member1
    ec.teamConfigs = [(1, 1)]  # single 1v1 config -> 2 active slots, both owned per-episode
    ec.teamWeights = [1.0]
    ec.randomSeed = 123

    cfg = L.LearnerConfig()
    cfg.deviceType = L.LearnerDeviceType.CPU  # bisect: is CUDA model construction the crash?
    # Absolute mesh path so the learner's RocketSim::Init resolves regardless of the host CWD.
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
    cfg.ppo.lambdaFSP = 0.0  # smoke test: no FSP head

    # Two members (same superset width) to exercise multi-member routing.
    m0 = make_perceiver(39)
    m1 = make_perceiver(39)
    cfg.memberPerceiver = [m0, m1]

    learner = L.create_learner(ec, cfg)
    print("obs_size:", learner.obs_size)
    print("action_amount:", learner.action_amount)
    # Superset layout: query(208) + tokens(151)*kv(69) + mask(151) = 10778.
    assert learner.obs_size == 208 + 151 * 69 + 151, f"unexpected obs_size {learner.obs_size}"

    learner.start_agents()
    reports = learner.run_iteration()
    learner.stop_agents()

    print("type(reports):", type(reports).__name__)
    print("len(reports):", len(reports))
    assert isinstance(reports, list), "run_iteration must return a list"
    assert len(reports) == 2, f"expected 2 member reports, got {len(reports)}"
    for i, r in enumerate(reports):
        d = r.as_dict() if hasattr(r, "as_dict") else r
        print(f"member {i} report keys:", list(d.keys())[:6])
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
