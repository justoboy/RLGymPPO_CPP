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
    # Native TemporalObsBuilder layout (maxPlayers=4 -> but teamSize=1 so nBlue/nOrange=1):
    # the builder's native widths are query=59, kv=30, tokens=559 (see examplemain.cpp comment).
    # For the smoke test both members use the same (native) width; the multi-member plumbing is
    # exercised by having 2 policies + 2 buffers + slotOwners routing, not by width divergence.
    pc.entityFeatures = ef
    pc.queryFeatures = 59
    pc.kvFeatures = 30
    pc.nTokens = 559
    return pc


def main():
    print("CWD:", os.getcwd())
    print("collision_meshes exists:", os.path.isdir(os.path.join(os.getcwd(), "collision_meshes")))
    ec = L.EnvConfig()
    ec.teamSize = 1
    ec.spawnOpponents = True
    ec.gameMode = L.GameMode.SOCCAR
    ec.nControls = 9
    ec.entityFeatures = 30  # native builder width

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

    # Two members (same native width) to exercise the multi-member plumbing.
    m0 = make_perceiver(30)
    m1 = make_perceiver(30)
    cfg.memberPerceiver = [m0, m1]
    # 2 slots: slot 0 -> member 0 (blue), slot 1 -> member 1 (orange).
    cfg.slotOwners = [0, 1]

    learner = L.create_learner(ec, cfg)
    print("obs_size:", learner.obs_size)
    print("action_amount:", learner.action_amount)
    assert learner.obs_size == 59 + 559 * 30 + 559, f"unexpected obs_size {learner.obs_size}"

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
