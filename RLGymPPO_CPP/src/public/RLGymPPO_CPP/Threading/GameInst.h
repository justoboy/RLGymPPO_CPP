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

		// The obs of the state stepped into on the most recent Step (the DONE-state obs when that step
		// ended the episode). curObs is overwritten with the NEXT episode's reset obs on a done, so the
		// trajectory's next_obs must be read from HERE, not curObs: the append loop indexes it by the OLD
		// episode's player count (prevOwners), and a count that SHRINKS at the reset (e.g. a 1v1 touch
		// episode -> a 1v0 goal episode) would otherwise index the post-reset tensor out of range.
		FList2 lastStepObs;

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

	// Per-slot running episode return (sum of that slot's own reward over the current episode). Kept
	// SEPARATE per slot (never summed across slots) so a self-play episode's two cars keep their own
	// returns; the per-owner reward is the MAX over that owner's slots (the better car's return), so it
	// never cancels to zero the way summing both teams' symmetric rewards did.
	std::vector<float> curSlotRew;

	// Per-owner (member id) tallies accumulated across every episode completed this iteration. GetOwnerMetrics
	// averages these into reports[owner]["goals_for_mean"] / ["touches_for_mean"] / ["Average Episode Reward"]
	// for the metric-gated curriculum driver to read. In self-play (one member owns both cars) the goals/
	// touches trackers average across both cars (the policy's true rate) and the reward tracker takes the
	// max of the two cars' returns (the better car's return, so the reward graph climbs instead of flip-flopping).
	std::vector<AvgTracker> ownerGoals, ownerTouches, ownerEpRew;

	uint64_t totalSteps;

	AvgTracker avgStepRew;

	// Will be reset every iteration, when ResetMetrics() is called
	Report _metrics = {};

	StepCallback stepCallback = NULL;

	// NOTE: Gym and match will be deleted when GameInst is deleted
	GameInst(RLGSC::Gym* gym, RLGSC::Match* match) : gym(gym), match(match) {
		totalSteps = 0;
		curTouches.assign(MAX_SLOTS, 0);
		curSlotRew.assign(MAX_SLOTS, 0.0f);
		ownerGoals.resize(MAX_SLOTS);
		ownerTouches.resize(MAX_SLOTS);
		ownerEpRew.resize(MAX_SLOTS);
	}

	RG_NO_COPY(GameInst);

	void ResetMetrics() {
		avgStepRew.Reset();
		_metrics.Clear();
		for (auto& t : ownerGoals) t.Reset();
		for (auto& t : ownerTouches) t.Reset();
		for (auto& t : ownerEpRew) t.Reset();
		std::fill(curTouches.begin(), curTouches.end(), 0);
		std::fill(curSlotRew.begin(), curSlotRew.end(), 0.0f);
	}

	void Start();
	RLGSC::Gym::StepResult Step(const FList& actions);

		~GameInst() {
			delete gym;
			delete match;
		}
	};
}