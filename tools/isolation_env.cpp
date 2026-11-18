// Minimal isolation test: does the live Gym env Reset() work at all in this environment?
// No torch, no pybind, no Learner. Just RocketSim + Match + Gym + Reset.
#include <RLGymSim_CPP/Envs/Match.h>
#include <RLGymSim_CPP/Gym.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CommonRewards.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CombinedReward.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/NoTouchCondition.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>
#include <RLGymSim_CPP/Utils/OBSBuilders/TemporalObsBuilder.h>
#include <RLGymSim_CPP/Utils/StateSetters/RandomState.h>
#include <RLGymSim_CPP/Utils/ActionParsers/ContinuousActionParser.h>
#include <cstdio>

using namespace RLGSC;

int main() {
	printf("init rocket sim...\n"); fflush(stdout);
	RocketSim::Init("collision_meshes", true);

	printf("build match...\n"); fflush(stdout);
	auto rewards = new CombinedReward({
		{ new FaceBallReward(), 0.1f },
		{ new VelocityPlayerToBallReward(), 0.5f },
		{ new VelocityBallToGoalReward(), 1.0f },
		{ new EventReward({.teamGoal = 1.f, .concede = -1.f}), 50.f },
	});
	std::vector<TerminalCondition*> tcs = {
		new NoTouchCondition(10.f * 120 / 8),
		new GoalScoreCondition()
	};
	auto obs = new TemporalObsBuilder();
	obs->activeMode = GameMode::SOCCAR;
	auto ap = new ContinuousActionParser(8);

	// Random-kickoff resetter (the production factory path). Mode is fixed at Arena::Create time.
	auto stateSetter = new RandomState(true, true, true);

	Match* match = new Match(rewards, tcs, obs, ap, stateSetter, 1, true);

	printf("build gym...\n"); fflush(stdout);
	Gym* gym = new Gym(match, 8, CAR_CONFIG_OCTANE, GameMode::SOCCAR, MutatorConfig(GameMode::SOCCAR));

	printf("reset...\n"); fflush(stdout);
	auto obsSet = gym->Reset();
	printf("obsSet.size()=%zu\n", obsSet.size()); fflush(stdout);
	if (!obsSet.empty())
		printf("obs[0].size()=%zu\n", obsSet[0].size()); fflush(stdout);

	printf("ISOLATION OK\n"); fflush(stdout);
	return 0;
}
