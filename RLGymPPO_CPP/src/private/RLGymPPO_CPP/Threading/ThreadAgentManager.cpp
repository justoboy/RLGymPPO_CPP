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
		// Each agent has already routed every COMPLETED episode's per-slot trajectory into its per-owner
		// ownerTrajectories (inside _RunFunc at each done). Flush the still-in-flight (cut-short) episodes
		// too, then concatenate each owner's accumulated trajectory into the result for that owner.
		for (auto agent : agents) {
			agent->trajMutex.lock();
			agent->FlushInFlightLocked();

			for (int g = 0; g < (int)agent->ownerTrajectories.size(); g++) {
				auto& traj = agent->ownerTrajectories[g];
				if (traj.size > 0) {
					if (results[g].size == 0)
						results[g] = traj;
					else
						results[g].Append(traj);
					totals[g] += traj.size;
					traj.Clear();
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
	// Per-step mean reward across active slots (diagnostic). The per-owner episode reward is NOT summed
	// here (that cross-team sum was what cancelled to zero); it is routed per-owner in GetOwnerMetrics.
	AvgTracker avgStepRew;
	for (auto agent : agents) {
		for (auto game : agent->gameInsts) {
			avgStepRew += game->avgStepRew;
		}
	}
	
	report["Average Step Reward"] = avgStepRew.Get();

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

void RLGPC::ThreadAgentManager::GetOwnerMetrics(std::vector<Report>& reports) {
	// One accumulator per owner (member). Each game accumulated per-owner tallies across the episodes it
	// completed this iteration; sum them per owner and average (AvgTracker::Get returns NAN if no episodes).
	// epRew is the mean of each owner's per-episode max-slot return (the better car's return in self-play),
	// so the reported reward climbs with skill instead of cancelling across the two symmetric teams.
	std::vector<AvgTracker> goals(policies.size()), touches(policies.size()), epRew(policies.size());

	for (auto agent : agents) {
		agent->gameStepMutex.lock();
		for (auto game : agent->gameInsts) {
			for (size_t g = 0; g < policies.size(); g++) {
				goals[g] += game->ownerGoals[g];
				touches[g] += game->ownerTouches[g];
				epRew[g] += game->ownerEpRew[g];
			}
		}
		agent->gameStepMutex.unlock();
	}

	for (size_t g = 0; g < policies.size() && g < reports.size(); g++) {
		reports[g]["goals_for_mean"] = goals[g].Get();
		reports[g]["touches_for_mean"] = touches[g].Get();
		reports[g]["Average Episode Reward"] = epRew[g].Get();
	}
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
