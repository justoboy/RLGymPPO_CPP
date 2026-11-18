"""Python league driver (P3-ML Step 5): the port of reference/main.py:run_epoch on top of the
generic pybind learner.

The C++ learner is a single fixed-topology, multi-member unit: it owns one policy per "member"
(bot) and a fixed slotOwners map (slot j -> member id) that is identical for every game, and
RunIteration() performs one full Jacobi epoch (freeze all -> collect -> update each member on its
own routed buffers). The league is composed in Python by configuring memberPerceiver (one per bot)
+ slotOwners and interleaving iterations. N=1 collapses to single-bot self-play.

WHY A SEPARATE DRIVER (not C++): the public C++ surface stays generic single-policy ("one learner,
one config, one iteration"); "league" is an emergent pattern of holding several members and routing
slots to owners. Keeping it in Python keeps the public repo generic.

TWO-INTERPRETER SPLIT: BC checkpoints are torch.save dicts produced by the py3.14 reference env;
the .pyd is built against anaconda py3.9. So this driver (run under anaconda 3.9, it imports the
.pyd) shells out to the venv python to run convert_bc_checkpoint.py, which writes the WEIG blobs
(PPO_POLICY.lt / PPO_CRITIC.lt / PPO_FSP.lt) AND a spec.json sidecar carrying each member's exact
Perceiver widths. The driver reads spec.json to build the matching PerceiverConfig WITHOUT importing
reference (earl_pytorch is py3.14-only).
"""
import argparse
import json
import os
import subprocess
import sys

# Native fallback layout (used only when a checkpoint carries no recorded spec, e.g. the synthetic
# smoke fixture). Mirrors build_seed_fixture.py + TemporalObsBuilder defaults.
NATIVE = dict(entity_features=30, query_features=59, kv_features=30, n_tokens=559,
              n_controls=9, head="continuous")

VENV_PY = os.environ.get("REFERENCE_VENV_PY", r"D:\RLBotTraining\reference\.venv\Scripts\python.exe")
TOOLS = os.path.dirname(os.path.abspath(__file__))


def _import_module(build_dir):
    """Make the .pyd importable: torch DLLs on the DLL search path, the build dir on sys.path, and
    torch imported BEFORE the extension (avoids C10 dispatcher double-registration)."""
    for p in (
        r"C:\Users\justo\anaconda3\lib\site-packages\torch\lib",
        r"C:\Users\justo\anaconda3\Library\bin",
        r"C:\Users\justo\anaconda3\DLLs",
    ):
        if os.path.isdir(p):
            os.add_dll_directory(p)
    sys.path.insert(0, build_dir)
    import torch  # noqa: F401  (must precede rlgym_learner import)
    import rlgym_learner as L
    return L


def convert_checkpoint(L_module_unused, pt_path, weig_out):
    """Run the venv converter to turn a BC .pt into a WEIG folder (+ spec.json sidecar)."""
    if not os.path.isfile(VENV_PY):
        raise SystemExit(f"venv python not found: {VENV_PY}")
    subprocess.run([VENV_PY, os.path.join(TOOLS, "convert_bc_checkpoint.py"), pt_path, weig_out],
                   check=True)
    if not os.path.isfile(os.path.join(weig_out, "PPO_POLICY.lt")):
        raise SystemExit(f"converter did not emit actor blob in {weig_out}")
    return weig_out


def read_spec(weig_folder):
    """Read the spec.json sidecar; fall back to the native layout when absent."""
    path = os.path.join(weig_folder, "spec.json")
    if not os.path.isfile(path):
        return dict(NATIVE)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_env_config(L, member_layouts, *, tick_skip=8, game_mode, team_size, spawn_opponents=True,
                     spawn_curriculum=False, spawn_ratio=1.0, spawn_prob=0.5, mutator_seed=0):
    """EnvConfig at the superset (widest) member layout. The env always emits the superset flat obs;
    the C++ threading layer column-projects each slot down to its owner's own width."""
    sup = max(member_layouts, key=lambda m: m["entity_features"])
    ec = L.EnvConfig()
    ec.tickSkip = tick_skip
    ec.gameMode = game_mode
    ec.teamSize = team_size
    ec.spawnOpponents = spawn_opponents
    ec.spawnCurriculumEnabled = bool(spawn_curriculum)
    ec.spawnCurriculumRatio = float(spawn_ratio)
    ec.spawnCurriculumProb = float(spawn_prob)
    ec.mutatorSeed = int(mutator_seed)
    ec.entityFeatures = int(sup["entity_features"])
    ec.queryFeatures = int(sup["query_features"])
    ec.kvFeatures = int(sup["kv_features"])
    ec.nTokens = int(sup["n_tokens"])
    ec.nControls = int(sup["n_controls"])
    return ec


def build_perceiver(L, layout, *, n_dims=256, n_heads=8, n_layers=1):
    pc = L.PerceiverConfig()
    pc.nDims = n_dims
    pc.nHeads = n_heads
    pc.nLayers = n_layers
    pc.nControls = int(layout["n_controls"])
    pc.headType = L.HeadType.DISCRETE if layout.get("head") == "discrete" else L.HeadType.CONTINUOUS
    pc.entityFeatures = int(layout["entity_features"])
    pc.queryFeatures = int(layout["query_features"])
    pc.kvFeatures = int(layout["kv_features"])
    pc.nTokens = int(layout["n_tokens"])
    return pc


class LeagueDriver:
    """Wraps one fixed-topology multi-member learner + its BC seeding + epoch stepping."""

    def __init__(self, L, *, bot_checkpoints, slot_owners, team_size, game_mode, seed,
                 save_dir, load_dir, num_threads=1, games_per_thread=1,
                 timesteps_per_iteration=64, exp_buffer_size=4096,
                 batch_size=32, mini_batch_size=16, epochs=1,
                 actor_lr=1e-4, critic_lr=1e-4, ent_coef=0.01, clip_range=0.2, lambda_fsp=0.0,
                 standardize_returns=False, device="CPU", spawn_curriculum=False):
        self.L = L
        self.learner = None
        self.member_weig = []

        # 1. Convert each bot's BC .pt -> WEIG folder (+ spec.json), collecting each member's layout.
        member_layouts = []
        for i, pt in enumerate(bot_checkpoints):
            weig = os.path.join(save_dir, f"_seed_member_{i}")
            os.makedirs(save_dir, exist_ok=True)
            convert_checkpoint(L, pt, weig)
            layout = read_spec(weig)
            member_layouts.append(layout)
            self.member_weig.append(weig)
            print(f"[driver] member {i}: layout {layout}")

        # 2. EnvConfig at the superset layout + one PerceiverConfig per member.
        ec = build_env_config(L, member_layouts, tick_skip=8, game_mode=game_mode,
                              team_size=team_size, spawn_curriculum=spawn_curriculum)
        cfg = L.LearnerConfig()
        cfg.deviceType = getattr(L.LearnerDeviceType, device)
        cfg.collisionMeshesFolder = os.environ.get(
            "COLLISION_MESHES", r"D:\RLBotTraining\RLGymSim_CPP\collision_meshes")
        cfg.numThreads = num_threads
        cfg.numGamesPerThread = games_per_thread
        cfg.timestepsPerIteration = timesteps_per_iteration
        cfg.expBufferSize = exp_buffer_size
        cfg.standardizeReturns = standardize_returns  # P6: raw returns to match Python
        cfg.checkpointSaveFolder = save_dir
        cfg.checkpointLoadFolder = load_dir or ""
        cfg.sendMetrics = False
        cfg.randomSeed = seed
        cfg.ppo.batchSize = batch_size
        cfg.ppo.miniBatchSize = mini_batch_size
        cfg.ppo.epochs = epochs
        cfg.ppo.policyLR = actor_lr
        cfg.ppo.criticLR = critic_lr
        cfg.ppo.entCoef = ent_coef
        cfg.ppo.clipRange = clip_range
        cfg.ppo.lambdaFSP = lambda_fsp
        cfg.memberPerceiver = [build_perceiver(L, m) for m in member_layouts]
        cfg.slotOwners = list(slot_owners)

        # 3. Create the learner, then seed every member from its BC WEIG folder.
        learner = L.create_learner(ec, cfg)
        for g, weig in enumerate(self.member_weig):
            learner.seed_member(g, weig)
            print(f"[driver] seeded member {g} from {weig}")
        self.learner = learner
        print(f"[driver] learner ready: obs_size={learner.obs_size} "
              f"action_amount={learner.action_amount} members={len(member_layouts)}")

    def run_epoch(self):
        """One Jacobi epoch: collect (all frozen) -> update each member on its routed buffers."""
        return self.learner.run_iteration()

    def start(self):
        self.learner.start_agents()

    def stop(self):
        self.learner.stop_agents()

    def save(self):
        self.learner.save()


def load_run_config(path):
    """Minimal YAML-ish loader mirroring reference/train_config.py's schema (we avoid a yaml dep in
    the py3.9 anaconda env by parsing the small flat schema by hand)."""
    import yaml  # pyyaml is available in anaconda base; if not, the driver is unusable anyway
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw


def main():
    ap = argparse.ArgumentParser(description="C++ learner league driver (P3-ML)")
    ap.add_argument("--config", required=True, help="run-config YAML (bots, slot_owners, ...)")
    ap.add_argument("--build-dir", required=True, help="dir containing rlgym_learner.pyd")
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    L = _import_module(args.build_dir)
    raw = load_run_config(args.config)

    bots = raw["bots"]  # list of {name, checkpoint} dicts
    bot_checkpoints = [b["checkpoint"] for b in bots]
    slot_owners = [int(x) for x in raw["slot_owners"]]
    team_size = int(raw.get("team_size", 1))
    game_mode = getattr(L.GameMode, str(raw.get("game_mode", "SOCCAR")))

    driver = LeagueDriver(
        L,
        bot_checkpoints=bot_checkpoints,
        slot_owners=slot_owners,
        team_size=team_size,
        game_mode=game_mode,
        seed=int(raw.get("seed", 123)),
        save_dir=raw["save_dir"],
        load_dir=raw.get("load_dir"),
        num_threads=int(raw.get("num_threads", 1)),
        games_per_thread=int(raw.get("games_per_thread", 1)),
        timesteps_per_iteration=int(raw.get("timesteps_per_iteration", 64)),
        batch_size=int(raw.get("batch_size", 32)),
        mini_batch_size=int(raw.get("mini_batch_size", 16)),
        epochs=int(raw.get("epochs_ppo", 1)),
        actor_lr=float(raw.get("actor_lr", 1e-4)),
        critic_lr=float(raw.get("critic_lr", 1e-4)),
        ent_coef=float(raw.get("ent_coef", 0.01)),
        lambda_fsp=float(raw.get("lambda_fsp", 0.0)),
        standardize_returns=bool(raw.get("standardize_returns", False)),
        device=str(raw.get("device", "CPU")),
        spawn_curriculum=bool(raw.get("spawn_curriculum", False)),
    )

    driver.start()
    try:
        for it in range(args.epochs):
            reports = driver.run_epoch()
            for g, r in enumerate(reports):
                d = r.as_dict() if hasattr(r, "as_dict") else r
                print(f"[epoch {it}] member {g} report keys: {list(d.keys())[:6]}")
            driver.save()
    finally:
        driver.stop()
    print("LEAGUE DRIVER DONE")


if __name__ == "__main__":
    main()
