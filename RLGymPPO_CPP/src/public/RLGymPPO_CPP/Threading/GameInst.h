#pragma once
#include "../Lists.h"
#include "../Util/AvgTracker.h"
#include "../Util/Report.h"

namespace RLGPC {
	typedef std::function<void(class GameInst*, const RLGSC::Gym::StepResult&, Report&)> StepCallback;

	// Environment creation func for each ThreadAgent
	struct EnvCreateResult {
		RLGSC::Match* match;
		RLGSC::Gym* gym;
	};
	typedef std::function<EnvCreateResult()> EnvCreateFn;

	class RG_IMEXPORT GameInst {
	public:

		// This game instance is being used for evaluation instead of training
		bool isEval = false;

		RLGSC::Gym* gym;
		RLGSC::Match* match;

		FList2 curObs;

		// Owner id per currently-active slot (blue-first order, same order as curObs rows /
		// gym->prevState.players). Re-read from gym->prevState.slotOwners on every episode reset, so the
	// learner routes each slot's trajectory to the member that owned it DURING that episode. For a
	// single-member self-play run this is all zeros (every slot owned by member 0).
	std::vector<int> curOwners;

	// The superset frame has 8 slots (2 * MAX_TEAM); owner ids are < memberCount <= 8, so all per-slot /
	// per-owner tally vectors below are fixed-width 8 and indexed directly by slot index / owner id.
	static constexpr int MAX_SLOTS = 8;

	// Per-slot touch count accumulated over the CURRENT episode (incremented on each step whose player
	// had ballTouchedStep). Goals are read cumulatively from the done-state's matchGoals.
	std::vector<int> curTouches;

	// Per-owner (member id) tallies accumulated across every episode completed this iteration. GetMetrics
	// averages these into reports[owner]["goals_for_mean"] / ["touches_for_mean"] for the metric-gated
	// curriculum driver to read (it gates on outcome tallies, NOT reward magnitude).
	std::vector<AvgTracker> ownerGoals, ownerTouches;

	uint64_t totalSteps;

	float curEpRew = 0;
	AvgTracker avgStepRew, avgEpRew;

	// Will be reset every iteration, when ResetMetrics() is called
	Report _metrics = {};

	StepCallback stepCallback = NULL;

	// NOTE: Gym and match will be deleted when GameInst is deleted
	GameInst(RLGSC::Gym* gym, RLGSC::Match* match) : gym(gym), match(match) {
		totalSteps = 0;
		curTouches.assign(MAX_SLOTS, 0);
		ownerGoals.resize(MAX_SLOTS);
		ownerTouches.resize(MAX_SLOTS);
	}

	RG_NO_COPY(GameInst);

	void ResetMetrics() {
		avgStepRew.Reset();
		avgEpRew.Reset();
		_metrics.Clear();
		for (auto& t : ownerGoals) t.Reset();
		for (auto& t : ownerTouches) t.Reset();
		std::fill(curTouches.begin(), curTouches.end(), 0);
	}

	void Start();
	RLGSC::Gym::StepResult Step(const FList& actions);

		~GameInst() {
			delete gym;
			delete match;
		}
	};
}