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
#include <RLGymSim_CPP/Utils/StateSetters/ModeSamplerMutator.h>
#include <RLGymSim_CPP/Utils/ActionParsers/ContinuousActionParser.h>

using namespace RLGSC;

namespace RLGPC {

// The arena's game mode is fixed at Arena::Create() time (Gym construction), so the league builds one
// env per mode rather than sampling per-episode. The StateSetter is either the plain random-kickoff
// resetter (single-bot self-play) or the per-episode active-set/matchup sampler (league mode).
static EnvCreateFn MakeEnvCreateFn(const EnvConfig& ec) {
	// A single shared counter (captured by shared_ptr so every copy of the lambda — one per ThreadAgent —
	// advances the same counter) gives every game across every agent a distinct sampler seed, so games'
	// matchup sampling stays independent instead of all drawing the same team-config sequence in lockstep.
	auto gameCounter = std::make_shared<int>(0);
	return [ec, gameCounter]() mutable -> EnvCreateResult {
		int seed = ec.randomSeed + (*gameCounter)++;
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

		// League mode (teamConfigs set): per-episode active-set spawner + matchup sampler that stamps the
		// per-slot owner map. Self-play (teamConfigs empty): random-kickoff setter that spawns teamSize
		// (+opponents) cars into the fresh empty arena; every slot is owned by member 0 (self-play).
		StateSetter* stateSetter;
		if (ec.teamConfigs.empty())
			stateSetter = new RandomState(true, true, true, ec.teamSize, ec.spawnOpponents, CAR_CONFIG_OCTANE);
		else
			stateSetter = new TeamConfigMutator(ec.teamConfigs, ec.teamWeights, ec.memberCount, ec.siblingFraction, (uint32_t)seed);

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
