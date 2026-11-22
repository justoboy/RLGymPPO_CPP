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

	{ // Accumulate each slot's OWN per-step reward into its running episode return (kept separate per
	  // slot, never summed across slots). avgStepRew is the per-step mean across active slots (a separate
	  // diagnostic); the per-owner episode return is derived from curSlotRew at episode end (max over the
	  // owner's slots), so a self-play episode's two symmetric cars no longer cancel to zero.
		int nActive = (int)stepResult.reward.size();
		float totalRew = 0;
		for (int j = 0; j < nActive; j++) {
			curSlotRew[j] += stepResult.reward[j];
			totalRew += stepResult.reward[j];
		}
		avgStepRew.Add(totalRew, nActive);
	}

	{ // Accumulate per-slot touch counts for the current episode (one increment per step the slot touched)
		int nActive = (int)stepResult.state.players.size();
		for (int j = 0; j < nActive; j++)
			if (stepResult.state.players[j].ballTouchedStep)
				curTouches[j] += 1;
	}

	if (stepCallback)
		stepCallback(this, stepResult, _metrics);

	// Environment ending
	if (stepResult.done) {
		// Route this episode's outcome tallies to each slot's owner. matchGoals is cumulative per episode
		// (cars are rebuilt each episode, so it starts at 0), curTouches is the per-slot touch count, and
		// curSlotRew is the per-slot episode return. For each owner we take the MAX over that owner's slots
		// (the better car's return) so the reported reward never cancels; goals/touches average across the
		// owner's slots (in self-play that is both cars -> the policy's true rate).
		int nActive = (int)stepResult.state.players.size();
		std::vector<float> best(MAX_SLOTS, 0.0f);
		std::vector<int> bestCount(MAX_SLOTS, 0);
		for (int j = 0; j < nActive; j++) {
			int owner = (j < (int)curOwners.size()) ? curOwners[j] : 0;
			if (owner < 0 || owner >= MAX_SLOTS) owner = 0;
			ownerGoals[owner].Add((float)stepResult.state.players[j].matchGoals);
			ownerTouches[owner].Add((float)curTouches[j]);
			// max over this owner's slots: first slot seen sets the value, later slots raise it.
			if (bestCount[owner] == 0) best[owner] = curSlotRew[j];
			else best[owner] = RS_MAX(best[owner], curSlotRew[j]);
			bestCount[owner]++;
			curTouches[j] = 0;  // reset this slot's touch counter for the next episode
			curSlotRew[j] = 0; // reset this slot's running return for the next episode
		}
		for (int o = 0; o < MAX_SLOTS; o++)
			if (bestCount[o] > 0)
				ownerEpRew[o].Add(best[o]);

		nextObs = gym->Reset();

		// New episode: the active set (and thus the per-slot owner map) may differ. Re-read it so the
		// learner routes the NEXT episode's slots to the right members.
		curOwners = ActiveOwners(gym->prevState);
	}

	curObs = nextObs;
	totalSteps++;

	return stepResult;
}