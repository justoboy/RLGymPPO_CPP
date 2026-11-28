#include "ThreadAgent.h"

#include "ThreadAgentManager.h"
#include <RLGymPPO_CPP/Util/Timer.h>

using namespace RLGPC;

// Build one contiguous (totalPlayers, obsSize) tensor by concatenating every game's per-player obs rows
// along dim 0 (row-major blocks == one row-major buffer of all players' obs; identical float32 to the old
// per-game-tensor + concat path, fewer dispatches). `obsField` selects WHICH obs to read: GameInst::curObs
// (the obs of the state each game is currently IN — the new episode after a done-reset) or
// GameInst::lastStepObs (the obs of the state each game just STEPPED into — the done-state obs, which
// curObs overwrites on a done). The trajectory's next_obs must use lastStepObs so its row count matches the
// OLD episode's player count (the append loop indexes it by prevOwners); using curObs there would index out
// of range whenever the active set SHRINKS at the reset (e.g. a 1v1 touch episode -> a 1v0 goal episode).
torch::Tensor MakeGamesOBSTensor(std::vector<GameInst*>& games, FList2 GameInst::*obsField) {
	assert(!games.empty());

	int totalPlayers = 0;
	for (auto game : games)
		totalPlayers += (int)(game->*obsField).size();
	int obsSize = (int)(games[0]->*obsField)[0].size();

	std::vector<float> buf;
	buf.reserve((size_t)totalPlayers * (size_t)obsSize);
	for (auto game : games)
		for (auto& row : game->*obsField)
			buf.insert(buf.end(), row.begin(), row.end());

	auto options = torch::TensorOptions().dtype(at::kFloat);
	return torch::from_blob(buf.data(), { (int64_t)totalPlayers, (int64_t)obsSize }, options).clone();
}

// Project one superset obs row to owner g's width. Identity (returns the row unchanged) when the
// owner's column gather is undefined (single-policy / superset owner).
static inline torch::Tensor ProjectRow(const torch::Tensor& fullRow, const torch::Tensor& colG) {
	if (!colG.defined())
		return fullRow;
	return fullRow.index_select(0, colG);
}

// Flush one game's per-slot trajectory buffers into the owner-routed accumulation buffers. For each slot
// that has data, mark truncation on its last step (a step that ended without a goal is truncated, not
// done) and append the whole per-slot trajectory end-to-end into ownerTrajs[owner]. `owners` is the owner
// map that was active during the episode those steps came from (constant within an episode). Slots past
// the active count have empty buffers and are skipped.
static void FlushSlots(
	std::vector<GameTrajectory>& perSlot, const std::vector<int>& owners,
	std::vector<GameTrajectory>& ownerTrajs
) {
	for (int j = 0; j < (int)perSlot.size(); j++) {
		auto& traj = perSlot[j];
		if (traj.size == 0)
			continue;

		// If the last timestep is not a done, mark it as truncated (the state stopped being continuous
		// because the data got cut short rather than because a goal was scored).
		traj.data.truncateds[traj.size - 1] = (traj.data.dones[traj.size - 1].item<float>() == 0);

		int owner = owners[j];
		if (ownerTrajs[owner].size == 0)
			ownerTrajs[owner] = traj;
		else
			ownerTrajs[owner].Append(traj);
		traj.Clear();
	}
}

void _RunFunc(ThreadAgent* ta) {
	RG_NOGRAD;
	ta->isRunning = true;

	auto mgr = (ThreadAgentManager*)ta->_manager;
	auto& games = ta->gameInsts;
	int numGames = ta->numGames;

	auto device = mgr->device;

	bool render = mgr->renderSender != NULL;
	if (render && mgr->renderDuringTraining) {
		if (ta->index != 0)
			render = false;
	}
	bool deterministic = mgr->deterministic;
	bool blockConcurrentInfer = mgr->blockConcurrentInfer;
	Timer stepTimer = {};

	// Start games (this also populates each game's curObs + curOwners for its first episode).
	for (auto game : games)
		game->Start();

	// Will stores our current observations for all our games
	torch::Tensor curObsTensor = MakeGamesOBSTensor(games, &GameInst::curObs);

	constexpr bool halfPrec = false;

	auto& policies = mgr->policies;
	auto& colIdx = mgr->colIdx;
	bool singlePolicy = mgr->singlePolicy;
	int numOwners = (int)policies.size();

	while (ta->shouldRun) {

		if (render)
			stepTimer.Reset();

		// Don't run if we reached our step limit. Both inner waits MUST also check shouldRun: once an
		// agent exceeds its per-agent budget the first condition is permanently true, so without the
		// shouldRun guard the worker parks there forever and Stop() (which busy-spins on isRunning)
		// never returns. This was the g32/t2 deadlock (0% CPU/GPU hang).
		while (ta->shouldRun && ta->stepsCollected > ta->maxCollect)
			THREAD_WAIT();
		if (!ta->shouldRun)
			break;

		while (ta->shouldRun && mgr->disableCollection)
			THREAD_WAIT();
		if (!ta->shouldRun)
			break;

		// Snapshot the owner map active during THIS step (the episode currently running). The active set
		// and owners can change at the next reset, so we freeze them for routing this step's data.
		std::vector<std::vector<int>> prevOwners(numGames);
		std::vector<int64_t> flatOwners;
		flatOwners.reserve(curObsTensor.size(0));
		for (int i = 0; i < numGames; i++) {
			prevOwners[i] = games[i]->curOwners;
			for (int o : prevOwners[i])
				flatOwners.push_back(o);
		}

		// Infer actions for all our agents in all our games.
		Timer policyInferTimer = {};

		if (blockConcurrentInfer)
			mgr->inferMutex.lock();

		int totalPlayers = (int)curObsTensor.size(0);
		int nControls = (int)policies[0]->config.nControls;

		GaussianPolicy::ActionResult actionResults;
		try {
			if (singlePolicy) {
				// Byte-identical to the historical single-policy path.
				torch::Tensor curObsTensorDevice = halfPrec
					? curObsTensor.to(RG_HALFPERC_TYPE).to(device, true)
					: curObsTensor.to(device, true);
				actionResults = policies[0]->GetAction(curObsTensorDevice, deterministic);
				if (halfPrec) {
					actionResults.action = actionResults.action.to(torch::ScalarType::Float);
					actionResults.logProb = actionResults.logProb.to(torch::ScalarType::Float);
				}
			} else {
				// Multi-policy: per-owner column-project the superset obs to that owner's width, run that
				// owner's policy on just its rows, and scatter the per-slot actions back into full tensors.
				// The owner->row mapping is rebuilt every step because the active set varies per episode.
				auto fullAction = torch::zeros({ (int64_t)totalPlayers, (int64_t)nControls }, torch::TensorOptions().dtype(at::kFloat));
				auto fullLogProb = torch::zeros({ (int64_t)totalPlayers }, torch::TensorOptions().dtype(at::kFloat));
				for (int g = 0; g < numOwners; g++) {
					std::vector<int64_t> rowsG;
					for (int r = 0; r < totalPlayers; r++)
						if (flatOwners[r] == g)
							rowsG.push_back(r);
					if (rowsG.empty())
						continue;
					auto rowsT = torch::from_blob(rowsG.data(), { (int64_t)rowsG.size() }, torch::TensorOptions().dtype(torch::kLong)).clone();
					torch::Tensor projG = colIdx[g].defined() ? curObsTensor.index_select(1, colIdx[g]) : curObsTensor;
					torch::Tensor ownerObs = projG.index_select(0, rowsT).to(device, true);
					auto res = policies[g]->GetAction(ownerObs, deterministic);
					fullAction.index_copy_(0, rowsT, res.action);
					fullLogProb.index_copy_(0, rowsT, res.logProb);
				}
				actionResults.action = fullAction;
				actionResults.logProb = fullLogProb;
			}
		} catch (std::exception& e) {
			RG_ERR_CLOSE("Exception during policy->GetAction(): " << e.what());
		}
		if (blockConcurrentInfer)
			mgr->inferMutex.unlock();

		float policyInferTime = policyInferTimer.Elapsed();
		ta->times.policyInferTime += policyInferTime;

		// Step the gym with the actions we got
		Timer gymStepTimer = {};
		ta->gameStepMutex.lock();
		auto stepResults = new RLGSC::Gym::StepResult[numGames];
		int actionsOffset = 0;
		bool anyReset = false; // did any game end its episode this step (curObs now holds the NEW episode)
		for (int i = 0; i < numGames; i++) {
			auto game = games[i];
			int numPlayers = (int)game->curObs.size(); // active count for this game this step

			// Actions output has a dimension for each player, but not for each game
			// So we will need to slice the section of it that is for this game
			auto actionSlice = actionResults.action.slice(0, actionsOffset, actionsOffset + numPlayers);

			// Flatten (numPlayers, nControls) -> flat FList for the continuous action parser
			stepResults[i] = game->Step(TENSOR_TO_FLIST(actionSlice.flatten()));
			anyReset |= stepResults[i].done;

			actionsOffset += numPlayers;
		}
		ta->gameStepMutex.unlock();

		// Make sure we got the end of actions
		// Otherwise there's a wrong number of actions for whatever reason
		assert(actionsOffset == actionResults.action.size(0));
		float envStepTime = gymStepTimer.Elapsed();
		ta->times.envStepTime += envStepTime;

		// The trajectory's next_obs: the obs of the state we just stepped into (the DONE-state obs when
		// this step ended the episode), read from lastStepObs. Its row count matches the OLD episode's
		// player count (prevOwners), which the append loop below indexes by. (curObs has already been
		// overwritten with the NEW episode's reset obs on a done, so reading curObs here would index out
		// of range whenever the active set shrinks at the reset.)
		torch::Tensor nextObsTensor = MakeGamesOBSTensor(games, &GameInst::lastStepObs);

		if (!render) {
			// Steps complete, add all timestep data to our per-slot trajectories, for each game. The stored
			// obs is the OWNER's projected width (identity in single-policy mode), so each member's
			// actor/critic/fsp see the width they were trained on (mirrors rollout.py:stored_obs).
			Timer trajAppendTimer = {};
			ta->trajMutex.lock();
			for (int i = 0, playerOffset = 0; i < numGames; i++) {
				int numPlayers = (int)prevOwners[i].size();

				auto& stepResult = stepResults[i];

				float done = (float)stepResult.done;
				float truncated = (float)false;

				auto tDone = torch::tensor(done);
				auto tTruncated = torch::tensor(truncated);

				for (int j = 0; j < numPlayers; j++) {
					int row = playerOffset + j;
					int owner = prevOwners[i][j];
					torch::Tensor storedCur = ProjectRow(curObsTensor[row], colIdx[owner]);
					torch::Tensor storedNext = ProjectRow(nextObsTensor[row], colIdx[owner]);

					ta->trajectories[i][j].AppendSingleStep(
						{
							storedCur,
							actionResults.action[row],
							actionResults.logProb[row],
							torch::tensor(stepResult.reward[j]),

#ifdef RG_PARANOID_MODE
							torch::Tensor(),
#endif

							storedNext,
							tDone,
							tTruncated
						}
					);
				}

				ta->stepsCollected += numPlayers;
				playerOffset += numPlayers;

				// Episode ended: route this episode's per-slot trajectories to their owners and clear, so
				// the next episode (which may have a different active set / owners) starts from empty.
				// A drill cut short by its time cap also emits done, so it counts as one completed
				// match for the match-based collection target (see ThreadAgent::episodesCompleted).
				if (stepResult.done) {
					ta->episodesCompleted.fetch_add(1);
					FlushSlots(ta->trajectories[i], prevOwners[i], ta->ownerTrajectories);
				}
			}
			ta->trajMutex.unlock();
			ta->times.trajAppendTime += trajAppendTimer.Elapsed();
		} else {
			// Update renderer
			auto renderSender = mgr->renderSender;
			auto renderGame = games[0];
			renderSender->Send(renderGame->gym->prevState, renderGame->gym->match->prevActions);

			// Delay for render
			// TODO: Somewhat dumb system using static variables
			{
				namespace chr = std::chrono;
				static auto lastRenderTime = chr::high_resolution_clock::now();
				auto durationSince = chr::high_resolution_clock::now() - lastRenderTime;
				lastRenderTime = chr::high_resolution_clock::now();

				int64_t micsSince = chr::duration_cast<chr::microseconds>(durationSince).count();

				double timeTaken = stepTimer.Elapsed();
				double targetTime = (1 / 120.0) * renderGame->gym->tickSkip / mgr->renderTimeScale;
				double sleepTime = RS_MAX(targetTime - timeTaken, 0);
				int64_t sleepMics = (int64_t)(sleepTime * 1000.0 * 1000.0);

				std::this_thread::sleep_for(chr::microseconds(sleepMics));
			}
		}

		// Now that the step is done, our next OBS becomes our current. On a non-done step lastStepObs == the
		// post-step curObs, so nextObsTensor is already the correct next curObsTensor (reuse it, no rebuild).
		// On a done step curObs holds the NEW episode's reset obs (possibly a different player count), so
		// rebuild from curObs to keep the next iteration's inference rows aligned with the live games.
		curObsTensor = anyReset ? MakeGamesOBSTensor(games, &GameInst::curObs) : std::move(nextObsTensor);

		delete[] stepResults;
	}

	ta->isRunning = false;
}

RLGPC::ThreadAgent::ThreadAgent(void* manager, int numGames, uint64_t maxCollect, EnvCreateFn envCreateFn, int index)
	: _manager(manager), numGames(numGames), maxCollect(maxCollect), index(index) {

	int numOwners = (int)((ThreadAgentManager*)manager)->policies.size();
	ownerTrajectories.resize(numOwners);

	trajectories.resize(numGames);
	for (int i = 0; i < numGames; i++)
		trajectories[i].resize(MAX_SLOTS); // superset slot count; only active slots are appended to

	// Build the game instances (does not start them; _RunFunc calls Start()).
	for (int i = 0; i < numGames; i++) {
		auto envCreateResult = envCreateFn();
		gameInsts.push_back(new GameInst(envCreateResult.gym, envCreateResult.match));
	}
}

void RLGPC::ThreadAgent::Start() {
	this->shouldRun = true;
	this->thread = std::thread(_RunFunc, this);
	this->thread.detach();
}

// Flush every game's currently-in-flight per-slot trajectory buffers into ownerTrajectories. Called by
// CollectTimesteps after the workers have stopped, to capture the episode that was cut short when
// collection ended (the already-finished episodes were flushed inside _RunFunc at each done). The owner
// map for these in-flight steps is each game's current curOwners (constant during the in-flight episode).
void RLGPC::ThreadAgent::FlushInFlightLocked() {
	for (int i = 0; i < (int)gameInsts.size(); i++)
		FlushSlots(trajectories[i], gameInsts[i]->curOwners, ownerTrajectories);
}

void RLGPC::ThreadAgent::Stop() {
	this->shouldRun = false;

	// Wait for thread to stop runing
	// TODO: Lame solution
	while (isRunning)
		RG_SLEEP(1);
}
