"""End-to-end smoke test for the WEIG seed path (BC -> learner weight seeding).

This is a WIRING test, not a numerical gate. The numerical correctness of LoadStateDict/
ApplyStateDict is already gated by ParityGate (which loads the SAME blobs via the SAME code path).
What this exercises is the pybind -> Learner::SeedMember -> PPOLearner::SeedFrom chain: that
seed_member() exists, routes to the right member, reads the WEIG folder layout that
convert_bc_checkpoint.py writes, and that the seeded weights then drive a real collect+learn
iteration without throwing.

Two interpreters are involved because of an ABI split:
  - the synthetic BC checkpoint needs earl_pytorch, installed only in the reference venv (py3.14)
  - the .pyd is built against anaconda 3.9 and cannot be imported under 3.14
So this script (run under anaconda 3.9) shells out to the venv python to build + convert the
fixture, then loads the .pyd and seeds from the resulting WEIG folder.
"""
import os
import subprocess
import sys
import tempfile

VENV_PY = r"D:\RLBotTraining\reference\.venv\Scripts\python.exe"
TOOLS = os.path.dirname(os.path.abspath(__file__))

# Native layout (must match build_seed_fixture.py + TemporalObsBuilder defaults).
QUERY_FEATURES = 59
KV_FEATURES = 30
N_TOKENS = 559
ENTITY_FEATURES = 30
N_CONTROLS = 9

# torch DLLs must be resolvable before importing the extension.
for p in (
    r"C:\Users\justo\anaconda3\lib\site-packages\torch\lib",
    r"C:\Users\justo\anaconda3\Library\bin",
    r"C:\Users\justo\anaconda3\DLLs",
):
    if os.path.isdir(p):
        os.add_dll_directory(p)

sys.path.insert(0, r"D:\RLBotTraining\RLGymPPO_CPP\build\Release")
import torch  # noqa: E402,F401  (must precede rlgym_learner import)
import rlgym_learner as L  # noqa: E402


def build_fixture(tmp):
    """Use the venv (py3.14) interpreter to build the synthetic .pt and convert it to WEIG."""
    if not os.path.isfile(VENV_PY):
        raise SystemExit(f"venv python not found: {VENV_PY}")
    pt = os.path.join(tmp, "bc.pt")
    weig = os.path.join(tmp, "weig")
    subprocess.run([VENV_PY, os.path.join(TOOLS, "build_seed_fixture.py"), pt], check=True)
    subprocess.run([VENV_PY, os.path.join(TOOLS, "convert_bc_checkpoint.py"), pt, weig], check=True)
    assert os.path.isfile(os.path.join(weig, "PPO_POLICY.lt")), "converter did not emit actor blob"
    assert os.path.isfile(os.path.join(weig, "PPO_CRITIC.lt")), "converter did not emit critic blob"
    return weig


def main():
    print("CWD:", os.getcwd())
    tmp = tempfile.mkdtemp(prefix="seed_smoke_")
    weig = build_fixture(tmp)
    print("WEIG folder:", weig)

    ec = L.EnvConfig()
    ec.teamSize = 1
    ec.spawnOpponents = True
    ec.gameMode = L.GameMode.SOCCAR
    ec.nControls = N_CONTROLS
    ec.entityFeatures = ENTITY_FEATURES
    ec.queryFeatures = QUERY_FEATURES
    ec.kvFeatures = KV_FEATURES
    ec.nTokens = N_TOKENS

    cfg = L.LearnerConfig()
    cfg.deviceType = L.LearnerDeviceType.CPU
    cfg.collisionMeshesFolder = r"D:\RLBotTraining\RLGymSim_CPP\collision_meshes"
    cfg.numThreads = 1
    cfg.numGamesPerThread = 1
    cfg.timestepsPerIteration = 64
    cfg.expBufferSize = 4096
    cfg.standardizeReturns = False
    cfg.checkpointLoadFolder = ""
    cfg.checkpointSaveFolder = ""
    cfg.sendMetrics = False
    cfg.ppo.batchSize = 32
    cfg.ppo.miniBatchSize = 16
    cfg.ppo.epochs = 1
    cfg.ppo.lambdaFSP = 0.0

    def perceiver():
        pc = L.PerceiverConfig()
        pc.nDims = 256
        pc.nHeads = 8
        pc.nLayers = 1
        pc.nControls = N_CONTROLS
        pc.headType = L.HeadType.CONTINUOUS
        pc.entityFeatures = ENTITY_FEATURES
        pc.queryFeatures = QUERY_FEATURES
        pc.kvFeatures = KV_FEATURES
        pc.nTokens = N_TOKENS
        return pc

    cfg.memberPerceiver = [perceiver(), perceiver()]
    cfg.slotOwners = [0, 1]

    learner = L.create_learner(ec, cfg)
    print("obs_size:", learner.obs_size)
    assert learner.obs_size == QUERY_FEATURES + N_TOKENS * KV_FEATURES + N_TOKENS

    # The actual thing under test: seed each member from the WEIG folder.
    learner.seed_member(0, weig)
    learner.seed_member(1, weig)
    print("seed_member(0/1) OK")

    learner.start_agents()
    reports = learner.run_iteration()
    learner.stop_agents()

    assert isinstance(reports, list) and len(reports) == 2, f"expected 2 reports, got {reports!r}"
    for i, r in enumerate(reports):
        d = r.as_dict() if hasattr(r, "as_dict") else r
        print(f"member {i} report keys:", list(d.keys())[:6])
    print("SEED SMOKE PASS")


if __name__ == "__main__":
    main()
