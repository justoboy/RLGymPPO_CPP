#include "ThreadAgent.h"

#include "ThreadAgentManager.h"
#include <RLGymPPO_CPP/Util/Timer.h>

using namespace RLGPC;

torch::Tensor MakeGamesOBSTensor(std::vector<GameInst*>& games) {
	// Every game's obs rows share the same width (obsSize), and concatenating row-major
	// blocks along dim 0 is identical to one row-major buffer of all players' obs. So we
	// copy every player's obs into a single contiguous buffer and build ONE tensor, which
	// removes both the per-game tensor construction and the torch::concat dispatch. The
	// float32 values are identical (torch::equal == true); only the dispatch count changes.
	assert(!games.empty());

	int totalPlayers = 0;
	for (auto game : games)
		totalPlayers += (int)game->curObs.size();
	int obsSize = (int)games[0]->curObs[0].size();

	std::vector<float> buf;
	buf.reserve((size_t)totalPlayers * (size_t)obsSize);
	for (auto game : games)
		for (auto& row : game->curObs)
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

	// Start games
	for (auto game : games)
		game->Start();

	// Will stores our current observations for all our games
	torch::Tensor curObsTensor = MakeGamesOBSTensor(games);

	constexpr bool halfPrec = false;

	auto& policies = mgr->policies;
	auto& colIdx = mgr->colIdx;
	bool singlePolicy = mgr->singlePolicy;

	// Per-owner flat row index tensors (constant across steps). Built once. In single-policy mode
	// owner 0 owns every row (identity), so the gather/scatter is a no-op.
	std::vector<torch::Tensor> rowIdxT;
	if (!singlePolicy) {
		for (int g = 0; g < (int)policies.size(); g++) {
			auto rows = mgr->BuildRowIdx(g, numGames);
			rowIdxT.push_back(
				torch::from_blob(rows.data(), { (int64_t)rows.size() }, torch::TensorOptions().dtype(torch::kLong)).clone()
			);
		}
	}

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
				auto fullAction = torch::zeros({ (int64_t)totalPlayers, (int64_t)nControls }, torch::TensorOptions().dtype(at::kFloat));
				auto fullLogProb = torch::zeros({ (int64_t)totalPlayers }, torch::TensorOptions().dtype(at::kFloat));
				for (int g = 0; g < (int)policies.size(); g++) {
					torch::Tensor projG = colIdx[g].defined() ? curObsTensor.index_select(1, colIdx[g]) : curObsTensor;
					torch::Tensor ownerObs = projG.index_select(0, rowIdxT[g]).to(device, true);
					auto res = policies[g]->GetAction(ownerObs, deterministic);
					fullAction.index_copy_(0, rowIdxT[g], res.action);
					fullLogProb.index_copy_(0, rowIdxT[g], res.logProb);
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
		for (int i = 0; i < numGames; i++) {
			auto game = games[i];
			int numPlayers = game->match->playerAmount;

			// Actions output has a dimension for each player, but not for each game
			// So we will need to slice the section of it that is for this game
			auto actionSlice = actionResults.action.slice(0, actionsOffset, actionsOffset + numPlayers);

			// Flatten (numPlayers, nControls) -> flat FList for the continuous action parser
			stepResults[i] = game->Step(TENSOR_TO_FLIST(actionSlice.flatten()));

			actionsOffset += numPlayers;
		}
		ta->gameStepMutex.unlock();

		// Make sure we got the end of actions
		// Otherwise there's a wrong number of actions for whatever reason
		assert(actionsOffset == actionResults.action.size(0));
		float envStepTime = gymStepTimer.Elapsed();
		ta->times.envStepTime += envStepTime;

		// Update our tensor storing the next observation after the step, from each gym
		torch::Tensor nextObsTensor = MakeGamesOBSTensor(games);

		if (!render) {
			// Steps complete, add all timestep data to our trajectories, for each game. The stored
			// obs is the OWNER's projected width (identity in single-policy mode), so each member's
			// actor/critic/fsp see the width they were trained on (mirrors rollout.py:stored_obs).
			Timer trajAppendTimer = {};
			ta->trajMutex.lock();
			for (int i = 0, playerOffset = 0; i < numGames; i++) {
				int numPlayers = games[i]->match->playerAmount;

				auto& stepResult = stepResults[i];

				float done = (float)stepResult.done;
				float truncated = (float)false;

				auto tDone = torch::tensor(done);
				auto tTruncated = torch::tensor(truncated);

				for (int j = 0; j < numPlayers; j++) {
					int row = playerOffset + j;
					int owner = singlePolicy ? 0 : mgr->slotOwners[j];
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

		// Now that the step is done, our next OBS becomes our current
		curObsTensor = nextObsTensor;

		delete[] stepResults;
	}

	ta->isRunning = false;
}

RLGPC::ThreadAgent::ThreadAgent(void* manager, int numGames, uint64_t maxCollect, EnvCreateFn envCreateFn, int index)
	: _manager(manager), numGames(numGames), maxCollect(maxCollect), index(index) {

	trajectories.resize(numGames);
	for (int i = 0; i < numGames; i++) {
		auto envCreateResult = envCreateFn();
		gameInsts.push_back(new GameInst(envCreateResult.gym, envCreateResult.match));
		trajectories[i].resize(envCreateResult.match->playerAmount);
	}
}

void RLGPC::ThreadAgent::Start() {
	this->shouldRun = true;
	this->thread = std::thread(_RunFunc, this);
	this->thread.detach();
}

void RLGPC::ThreadAgent::Stop() {
	this->shouldRun = false;

	// Wait for thread to stop runing
	// TODO: Lame solution
	while (isRunning)
		RG_SLEEP(1);
}
