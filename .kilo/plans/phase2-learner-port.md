# P3-ML: Multi-policy match primitive (league execution in C++) — implementation plan

Status: DECIDED (Option A, user-confirmed). Execute in order; each numbered step is its own commit, revertible.

## Context

The C++ learner is single-policy end-to-end: `ThreadAgentManager::policy` is one `GaussianPolicy*`,
`ThreadAgent::_RunFunc` calls `policy->GetAction(allObs)` once for every slot of every game, and
`CollectTimesteps` concatenates all per-slot trajectories into ONE `GameTrajectory` for ONE
`ExperienceBuffer`/`PPOLearner`. The Python league (`reference/main.py:run_epoch`, `league.py`,
`rollout.py`) plays sibling mixed-lineage matches: every slot is owned by a different member with
its OWN obs width (`entity_features = 25 + (14 if profile else 0)` -> 39-wide main bot vs 25-wide
style bots), per-slot obs is column-projected to the owner's width before that owner's forward
(`rollout.py:project_obs`), and each slot's reward uses its owner's term weights
(`ComposedReward.set_matchup`). N independent single-policy learners CANNOT share one env — so the
C++ threading layer must become a multi-policy match primitive. Python still owns the league
(policy weights, owner map, weights tables, update order); C++ just executes a supplied matchup fast.

Decisions already settled (do not re-litigate):
- D1: generalize `ThreadAgentManager`/`ThreadAgent`/`Learner` to N policies + per-slot owner map +
  per-owner buffers. N=1 + identity projection collapses to current behavior.
- D2: obs is always the 39-wide superset; owners with narrower specs get a per-owner column gather
  (exact port of `project_obs` on the flat layout).
- D3: reward is owner-routed: port `ComposedReward` + all terms as C++ reward functions; per-slot
  weight table installed at reset.
- D4: Jacobi needs no new mechanism: collection already runs on frozen weights (`disableCollection`
  during learn); learn-per-owner runs after collect.
- D5: pybind surface stays generic ("members", "slotOwners") — the word "league" never appears in
  public C++/bindings.

## Step 1 — C++ reward port (sim repo `RLGymSim_CPP`)

- [ ] Port `reference/reward_terms.py` terms (Goal, WinProb, BallHalf, Proximity, Alignment,
      Boost, Demo, AngVel, Grass, TouchHeight, TouchAccel, FlipReset, ... all ~25) as C++
      `RewardFunction`-style term objects under `RLGymSim_CPP/src/RLGymSim_CPP/Reward/`.
- [ ] Port `reference/reward.py:ComposedReward`: holds the full term list + `slotOwners` +
      per-owner `{term: weight}` tables + `team_spirit`/`opponent_punish` blending;
      `GetRewards(agents, state, terminated, truncated)` returns per-player reward weighted by
      that slot's owner table (`set_matchup` equivalent installs the tables).
- [ ] Parity fixture: seeded `GameState` sequence -> C++ per-player rewards == Python
      `ComposedReward.get_rewards` within 1e-6 (fixture gen/compare scripts under `tools/`).
- [ ] Gate: fixture PASS; single-owner (all slots owner 0) path byte-identical to today's
      `CombinedReward` output on the existing fixtures.

## Step 2 — C++ threading layer: multi-policy routing

- [ ] `ThreadAgentManager.h/.cpp`: replace `GaussianPolicy* policy` with
      `std::vector<GaussianPolicy*> policies`; add `std::vector<int> slotOwners` (length ==
      playerAmount, same for every game) and `std::vector<ExperienceBuffer*> buffers` (one per
      owner). Constructor/factory signature updated accordingly.
- [ ] Precompute per-owner gather index tensors once (not per step):
      - `ownerRowIdx[g]`: flat row indices of every (game, slot) pair owned by g (row-major,
        game-major then slot, matching `MakeGamesOBSTensor` row order).
      - `ownerColIdx[ef]`: flat column indices projecting the 39-wide superset to owner width ef:
        q cols `[0:ef] ++ [39:208]`, then per token t: kv cols `208 + t*69 + [0:ef] ++ [39:69]`,
        then mask cols unchanged. For ef==39 this is identity — set an `isIdentity` flag and skip
        the gather.
- [ ] `ThreadAgent::_RunFunc`: build the full superset obs tensor exactly as today; per owner g:
      `ownerObs = fullObs.index_select(1, ownerColIdx[ef_g]).index_select(0, ownerRowIdx[g])` ->
      `policies[g]->GetAction(ownerObs)` -> scatter per-owner action/logProb rows back into the
      full (totalPlayers, nControls) action tensor via `index_copy_(0, ownerRowIdx[g], ...)`.
      Everything downstream of that (game stepping, per-slot trajectory append at
      `trajectories[i][j]`) stays byte-identical.
- [ ] `CollectTimesteps(amount)` returns `std::vector<GameTrajectory>` indexed by owner: for each
      finished per-slot trajectory, route to `result[slotOwners[j]]` (the per-game truncation fixup
      at the top of the loop stays).
- [ ] N=1 regression guard: with one policy + all-zero slotOwners, the two index_selects are
      identity and behavior is identical; assert via existing parity fixtures.

## Step 3 — C++ Learner: multi-member orchestration

- [ ] `Learner.h/.cpp`: `std::vector<PPOLearner*> ppos` (one per member; member 0's replaces the
      old `ppo`), `std::vector<ExperienceBuffer*>` per-owner buffers, per-owner
      `WelfordRunningStat` return stats (keep field name compatibility for stats JSON).
- [ ] `LearnerConfig`: add `std::vector<PerceiverConfig> memberPerceiver` (spec per member) and
      `std::vector<int> slotOwners`; `config.ppo` stays the shared PPO hyperparams; FSP config per
      member as today.
- [ ] `RunIteration()`: `CollectTimesteps` -> per owner g: `AddNewExperience(traj[g], buf[g])` ->
      `ppos[g]->Learn(buf[g], report[g])`. `save(ownerIdx)`/`load(ownerIdx)` per-member checkpoint
      folders; stats JSON per member.
- [ ] `RLBotClient`/`examplemain.cpp` single-bot path must keep compiling and behaving identically
      (single member, all-zero slotOwners).

## Step 4 — pybind surface (generic wording, no "league")

- [ ] `LearnerBindings.cpp`: extend `EnvConfig` with per-term reward weights + `slotOwners`;
      `create_learner` takes `std::vector<PerceiverConfig>` member specs; `run_iteration(handle)`
      returns `list[dict]` (one report per member); `save(handle, dir, memberIdx)` /
      `load(handle, dir, memberIdx)`.
- [ ] `tools/convert_bc_checkpoint.py`: unchanged (already converts per-member `.pt` -> `.lt`).

## Step 5 — Python league driver (private repo)

- [ ] Port `reference/main.py:run_epoch` on top of the pybind learner: load N members (BC `.pt` ->
      `.lt`), build `LearnerConfig` with `memberPerceiver` + `slotOwners`, run epochs
      (freeze-all is implicit), per-member checkpoint save/load. Matchmaking (`draw_sibling`
      cycling + shuffle, team sizes per mode) stays in Python and just rewrites `slotOwners` +
      reward tables between iterations.

## Step 6 — Validation gates (hard, per FEATURE_PORT_PLAN.md §1)

- [ ] Single-bot self-play (N=1) on the existing seeded fixture: run-identical metrics to pre-P3.
- [ ] 2-member sibling fixture: seeded small batch -> per-member gradient direction cosine sim >
      0.99 vs Python league (`tools/` fixture, mirrors `league.py:draw_sibling`).
- [ ] Reward parity fixture PASS (Step 1 gate).
- [ ] Only AFTER all gates pass: Section-3 speed benchmark (todos 13-14), single-bot config,
      5 warm + 5 measured, ratio table -> `PERF_PROPOSALS.md`.

## Risks / notes

- `GaussianPolicy::GetAction` batching: per-owner calls mean up to N small forwards per step
  instead of one big one; acceptable (bandwidth-bound per BOTTLENECK_ANALYSIS), measure in Step 6.
- `index_select` column gather on a 10778-wide row x ~151 rows per step: negligible vs sim step.
- Keep `RG_PARANOID_MODE` debugCounters path compiling through the trajectory rework.
- If any gate fails with unexplainable divergence: STOP and report, do not ship.
