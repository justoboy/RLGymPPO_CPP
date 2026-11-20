#include "GameInst.h"

// Owner map for the episode currently running in `gym`. If the active-set setter stamped per-slot
// owners (league/sibling mode) use them; otherwise (plain self-play with a non-stamping setter such as
// RandomState) every active slot belongs to member 0. Always returns a vector sized to the active count.
static std::vector<int> ActiveOwners(const RLGSC::GameState& state) {
	int n = (int)state.players.size();
	if ((int)state.slotOwners.size() == n)
		return state.slotOwners;
	return std::vector<int>(n, 0);
}

void RLGPC::GameInst::Start() {
	curObs = gym->Reset();
	curOwners = ActiveOwners(gym->prevState);
}

RLGSC::Gym::StepResult RLGPC::GameInst::Step(const FList& actions) {

	// Step with agent actions
	auto stepResult = gym->Step(actions);

	auto& nextObs = stepResult.obs;

	{ // Update avg rewards (over the active players this step; the active set can vary per episode)
		float totalRew = 0;
		int nActive = (int)stepResult.reward.size();
		for (int i = 0; i < nActive; i++)
			totalRew += stepResult.reward[i];

		avgStepRew.Add(totalRew, nActive);
		curEpRew += (nActive > 0) ? (totalRew / nActive) : 0.f;
	}

	if (stepCallback)
		stepCallback(this, stepResult, _metrics);

	// Environment ending
	if (stepResult.done) {
		nextObs = gym->Reset();

		// New episode: the active set (and thus the per-slot owner map) may differ. Re-read it so the
		// learner routes the NEXT episode's slots to the right members.
		curOwners = ActiveOwners(gym->prevState);

		avgEpRew += curEpRew;
		curEpRew = 0;
	}

	curObs = nextObs;
	totalSteps++;

	return stepResult;
}