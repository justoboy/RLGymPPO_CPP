#include "ThreadAgentManager.h"
#include <RLGymPPO_CPP/Util/Timer.h>

void RLGPC::ThreadAgentManager::CreateAgents(EnvCreateFn func, int amount, int gamesPerAgent) {
	for (int i = 0; i < amount; i++) {
		int numGames = gamesPerAgent;
		if (renderSender && renderDuringTraining) {
			if (i == 0)
				numGames = 1;
		}
		auto agent = new ThreadAgent(this, numGames, maxCollect / amount, func, i);
		agents.push_back(agent);

		// playerAmount (slots per game) is fixed across games; learn it from the first game we build.
		if (playerAmount == 0)
			playerAmount = (int)agent->gameInsts[0]->match->playerAmount;
	}
}

std::vector<RLGPC::GameTrajectory> RLGPC::ThreadAgentManager::CollectTimesteps(uint64_t amount) {

	RG_LOG("Collecting timesteps...");
	// We will just wait in this loop until our agents have collected enough total timesteps
	while (true) {
		uint64_t totalSteps = 0;
		for (auto agent : agents)
			totalSteps += agent->stepsCollected;

		if (totalSteps >= amount)
			break;

		// "waiter! waiter! more timesteps please!"

		// TODO: Possibly sub-optimal waiting solution...
		// Could also just have the agents keep track of total step collection and unlock this thread?
		THREAD_WAIT();
	}

	// Our agents have collected the timesteps we need
	 
	RG_LOG("Concatenating timesteps...");

	// One trajectory per owner (member). In single-policy mode there is exactly one owner (index 0).
	std::vector<GameTrajectory> results(policies.size());
	std::vector<size_t> totals(policies.size(), 0);

	try {
		// Combine each owner's per-slot trajectories into that owner's long trajectory.
		for (auto agent : agents) {
			agent->trajMutex.lock();
			for (int i = 0; i < (int)agent->trajectories.size(); i++) {
				auto& trajSet = agent->trajectories[i];
				for (int j = 0; j < (int)trajSet.size(); j++) {
					auto& traj = trajSet[j];
					if (traj.size > 0) {
						// If the last timestep is not a done, mark it as truncated
						// The GAE needs to know when the environment state stops being continuous
						// This happens either because the environment reset (i.e. goal scored), called "done",
						//	or the data got cut short, called "truncated"
						traj.data.truncateds[traj.size - 1] = (traj.data.dones[traj.size - 1].item<float>() == 0);

						int owner = singlePolicy ? 0 : slotOwners[j];
						if (results[owner].size == 0)
							results[owner] = traj;
						else
							results[owner].Append(traj);
						totals[owner] += traj.size;
						traj.Clear();
					} else {
						// Kinda lame but does happen
					}
				}
			}
			agent->stepsCollected = 0;
			agent->trajMutex.unlock();
		}
	} catch (std::exception& e) {
		RG_ERR_CLOSE("Exception concatenating timesteps: " << e.what());
	}

	// Being extra paranoid in case something goes wrong
	size_t grandTotal = 0;
	for (size_t t : totals)
		grandTotal += t;
	if (grandTotal != amount)
		RG_LOG("ThreadAgentManager::CollectTimesteps(): collected " << grandTotal << " timesteps (target " << amount << ")");

	lastIterationTime = iterationTimer.Elapsed();
	iterationTimer.Reset();
	return results;
}

void RLGPC::ThreadAgentManager::GetMetrics(Report& report) {
	AvgTracker avgStepRew, avgEpRew;
	for (auto agent : agents) {
		for (auto game : agent->gameInsts) {
			avgStepRew += game->avgStepRew;
			avgEpRew += game->avgEpRew;
		}
	}
	
	report["Average Step Reward"] = avgStepRew.Get();
	report["Average Episode Reward"] = avgEpRew.Get();

	ThreadAgent::Times avgTimes = {};

	for (ThreadAgent* agent : agents)
		for (auto itr1 = avgTimes.begin(), itr2 = agent->times.begin(); itr1 != avgTimes.end(); itr1++, itr2++)
			*itr1 += *itr2;
	
	for (double& time : avgTimes)
		time /= agents.size();

	report["Env Step Time"] = avgTimes.envStepTime;
	// NOTE: Because of non-blocking mode, a good portion of policy inference time is waited when appending trajectories
	//	This means the trajectory append time is not correct at all, so this is a temporary solution
	report["Policy Infer Time"] = avgTimes.policyInferTime + avgTimes.trajAppendTime;
}

void RLGPC::ThreadAgentManager::ResetMetrics() {
	for (auto agent : agents) {
		agent->times = {};
		agent->gameStepMutex.lock();
		for (auto game : agent->gameInsts)
			game->ResetMetrics();
		agent->gameStepMutex.unlock();
	}
}
