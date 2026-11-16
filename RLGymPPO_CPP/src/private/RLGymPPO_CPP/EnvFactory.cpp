// DLL-side implementation of the env factory + learner factory. Compiled into RLGymPPO_CPP.dll so
// that the lambda's `new Gym(...)`/`new Match(...)` resolve to the DLL's (initialized) copy of the
// statically-linked RLGymSim_CPP state. See EnvFactory.h for the full rationale.
#include <RLGymPPO_CPP/EnvFactory.h>

#include <RLGymSim_CPP/Utils/RewardFunctions/CommonRewards.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CombinedReward.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/NoTouchCondition.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>
#include <RLGymSim_CPP/Utils/OBSBuilders/TemporalObsBuilder.h>
#include <RLGymSim_CPP/Utils/StateSetters/ModeSamplerMutator.h>
#include <RLGymSim_CPP/Utils/ActionParsers/ContinuousActionParser.h>

using namespace RLGSC;

namespace RLGPC {

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

		auto obs = new TemporalObsBuilder();
		obs->activeMode = ec.gameMode;
		obs->nBlue = ec.teamSize;
		obs->nOrange = ec.spawnOpponents ? ec.teamSize : 0;
		auto actionParser = new ContinuousActionParser(ec.tickSkip);

		// Chain: kickoff (mode/team) -> optional custom-spawn overlay. Each setter owns its config by
		// value, so nothing dangles after this lambda returns.
		auto teamSetter = new TeamConfigMutator(ec.mutatorSeed);
		auto modeSetter = new ModeSamplerMutator(obs, ec.mutatorSeed + 1u);
		auto spawnSetter = new SpawnCurriculumMutator(
			ec.spawnCurriculumEnabled, ec.spawnCurriculumRatio, ec.spawnCurriculumProb, ec.mutatorSeed + 2u);

		auto sequence = new MutatorSequence(std::vector<StateSetter*>{ teamSetter, modeSetter, spawnSetter });

		Match* match = new Match(
			rewards,
			terminalConditions,
			obs,
			actionParser,
			sequence,
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
