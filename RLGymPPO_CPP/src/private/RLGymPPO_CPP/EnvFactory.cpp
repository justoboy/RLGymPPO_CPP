// DLL-side implementation of the env factory + learner factory. Compiled into RLGymPPO_CPP.dll so
// that the lambda's `new Gym(...)`/`new Match(...)` resolve to the DLL's (initialized) copy of the
// statically-linked RLGymSim_CPP state. See EnvFactory.h for the full rationale.
#include <RLGymPPO_CPP/EnvFactory.h>

#include <RLGymSim_CPP/Utils/RewardFunctions/CommonRewards.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CombinedReward.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/NoTouchCondition.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>
#include <RLGymSim_CPP/Utils/OBSBuilders/TemporalObsBuilder.h>
#include <RLGymSim_CPP/Utils/StateSetters/RandomState.h>
#include <RLGymSim_CPP/Utils/ActionParsers/ContinuousActionParser.h>

using namespace RLGSC;

namespace RLGPC {

// The arena's game mode is fixed at Arena::Create() time (Gym construction), so the league builds one
// env per mode rather than sampling per-episode. The single StateSetter is the random-kickoff resetter.
static EnvCreateFn MakeEnvCreateFn(const EnvConfig& ec) {
	return [ec]() -> EnvCreateResult {
		auto rewards = new CombinedReward(
			{
				{ new FaceBallReward(), ec.faceBallWeight },
				{ new VelocityPlayerToBallReward(), ec.velPlayerToBallWeight },
				{ new VelocityBallToGoalReward(), ec.velBallToGoalWeight },
				{ new EventReward({ .teamGoal = ec.eventTeamGoal, .concede = ec.eventConcede }), ec.eventWeight },
			}
		);

		std::vector<TerminalCondition*> terminalConditions = {
			new NoTouchCondition(ec.noTouchTimeoutSecs * 120 / ec.tickSkip),
			new GoalScoreCondition()
		};

		// Superset temporal obs builder: emits q(208) ++ kv(151*69) ++ m(151) per agent.
		auto obs = new TemporalObsBuilder();
		obs->activeMode = ec.gameMode;
		obs->tickSkip = ec.tickSkip;

		auto actionParser = new ContinuousActionParser(ec.tickSkip);
		auto stateSetter = new RandomState(true, true, true);

		Match* match = new Match(
			rewards,
			terminalConditions,
			obs,
			actionParser,
			stateSetter,
			ec.teamSize,
			ec.spawnOpponents
		);

		Gym* gym = new Gym(match, ec.tickSkip, CAR_CONFIG_OCTANE, ec.gameMode, MutatorConfig(ec.gameMode));
		return { match, gym };
	};
}

Learner* CreateLearner(const EnvConfig& ec, const LearnerConfig& cfg) {
	// Keep the perceiver layout in the learner config aligned with the env config.
	LearnerConfig out = cfg;
	out.ppo.perceiver.queryFeatures = ec.queryFeatures;
	out.ppo.perceiver.kvFeatures = ec.kvFeatures;
	out.ppo.perceiver.nTokens = ec.nTokens;
	out.ppo.perceiver.nControls = ec.nControls;
	out.ppo.perceiver.entityFeatures = ec.entityFeatures;
	return new Learner(MakeEnvCreateFn(ec), out);
}

}
