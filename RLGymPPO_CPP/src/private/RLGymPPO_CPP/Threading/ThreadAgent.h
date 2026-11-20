#pragma once
#include "../PPO/GaussianPolicy.h"
#include <RLGymPPO_CPP/Threading/GameInst.h>
#include "GameTrajectory.h"

namespace RLGPC {
	class ThreadAgent {
	public:

		void* _manager;
		std::thread thread;
		int index;

		int numGames;
		std::vector<GameInst*> gameInsts;

		// Atomic: written by the main thread in Stop(), read by the worker in _RunFunc. Must be atomic so
		// the worker observes the shutdown flag promptly (a stale read here previously caused a deadlock:
		// once a worker parked in an inner wait loop it never re-read shouldRun, so Stop() spun forever).
		std::atomic<bool> shouldRun = false;
		std::atomic<bool> isRunning = false;

		struct Times {
			double
				envStepTime = 0,
				policyInferTime = 0,
				trajAppendTime = 0;

			double* begin() {
				return &envStepTime;
			}

			double* end() {
				return &trajAppendTime + 1;
			}
		};
		Times times = {}; // TODO: Convert to use Report instead

		// Per-slot trajectory buffers, indexed [game][slot]. The active set (and thus the number of
		// live slots) can vary per episode, so each game's slot vector is sized to the superset frame
		// width (MAX_SLOTS); only slots below the current episode's active count are appended to. When a
		// game's episode ends (done), that episode's per-slot buffers are flushed (end-to-end) into
		// ownerTrajectories[owner] and cleared, so a slot's data is always routed to the member that
		// owned it DURING that episode (the owner map is constant within an episode).
		static constexpr int MAX_SLOTS = 8; // superset car-slot count (2 * MAX_TEAM)
		std::vector<std::vector<GameTrajectory>> trajectories = {};

		// Accumulated, already-owner-routed experience for this agent, one buffer per member (owner id).
		std::vector<GameTrajectory> ownerTrajectories = {};

		std::atomic<uint64_t> stepsCollected = 0;
		uint64_t maxCollect;
		
		// Lock to prevent game stepping
		std::mutex gameStepMutex = {};

		// Lock to modify trajectories
		std::mutex trajMutex = {};

		ThreadAgent(void* manager, int numGames, uint64_t maxCollect, EnvCreateFn envCreateFn, int index);

		RG_NO_COPY(ThreadAgent);

		void Start();
		void Stop();

		// Flush the per-slot trajectory buffers of every game that is currently mid-episode (i.e. the
		// episode that was cut short when collection stopped) into ownerTrajectories, marking each cut
		// episode's last step truncated. Caller MUST hold trajMutex. Completes the owner-routing that
		// _RunFunc started for already-finished episodes.
		void FlushInFlightLocked();

		~ThreadAgent() {
			for (auto g : gameInsts)
				delete g;
		}
	};
}