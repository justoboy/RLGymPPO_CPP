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

		uint64_t totalSteps;

		float curEpRew = 0;
		AvgTracker avgStepRew, avgEpRew;

		// Will be reset every iteration, when ResetMetrics() is called
		Report _metrics = {};

		StepCallback stepCallback = NULL;

		// NOTE: Gym and match will be deleted when GameInst is deleted
		GameInst(RLGSC::Gym* gym, RLGSC::Match* match) : gym(gym), match(match) {
			totalSteps = 0;
		}

		RG_NO_COPY(GameInst);

		void ResetMetrics() {
			avgStepRew.Reset();
			avgEpRew.Reset();
			_metrics.Clear();
		}

	void Start();
	RLGSC::Gym::StepResult Step(const FList& actions);

		~GameInst() {
			delete gym;
			delete match;
		}
	};
}