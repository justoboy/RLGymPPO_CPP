// pybind11 extension module exposing the SINGLE-policy C++ learner to Python.
//
// Privacy/design note: this surface is deliberately GENERIC. It exposes one learner, one config,
// one env factory, and a single collect+learn iteration. It never mentions "league", "N bots", or
// "matchup routing" — those are emergent patterns from a Python driver that instantiates several of
// these and interleaves their RunIteration() calls (Jacobi: freeze all -> collect -> update each).
// Keeping the public surface single-policy keeps the public repo generic.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <RLGymPPO_CPP/Learner.h>
#include <RLGymPPO_CPP/LearnerConfig.h>
#include <RLGymPPO_CPP/PPO/PPOLearnerConfig.h>

#include <RLGymSim_CPP/Utils/RewardFunctions/CommonRewards.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CombinedReward.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/NoTouchCondition.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>
#include <RLGymSim_CPP/Utils/OBSBuilders/TemporalObsBuilder.h>
#include <RLGymSim_CPP/Utils/StateSetters/RandomState.h>
#include <RLGymSim_CPP/Utils/StateSetters/ModeSamplerMutator.h>
#include <RLGymSim_CPP/Utils/ActionParsers/ContinuousActionParser.h>

namespace py = pybind11;
using namespace RLGPC;
using namespace RLGSC;

// Everything the C++ env factory needs to build a Match+Gym, parameterized so Python configures it
// without touching the heavy sim types. Mirrors examplemain.cpp's EnvCreateFunc, extended with the
// P4 mode/team/spawn-curriculum knobs.
struct EnvConfig {
	int tickSkip = 8;
	float noTouchTimeoutSecs = 10.f;

	// Reward weights (CombinedReward). Mirrors the example's { FaceBall, VelPlayerToBall,
	// VelBallToGoal, EventReward{teamGoal,concede} }.
	float faceBallWeight = 0.1f;
	float velPlayerToBallWeight = 0.5f;
	float velBallToGoalWeight = 1.0f;
	float eventTeamGoal = 1.f;
	float eventConcede = -1.f;
	float eventWeight = 50.f;

	int teamSize = 1;
	bool spawnOpponents = true;

	// Fixed arena mode for this env (the league builds one env per mode). Maps to GameMode.
	GameMode gameMode = GameMode::SOCCAR;

	// Spawn-curriculum overlay (SpawnCurriculumMutator). Disabled by default -> native kickoff only.
	bool spawnCurriculumEnabled = false;
	float spawnCurriculumRatio = 1.0f;
	float spawnCurriculumProb = 0.5f;

	// Per-episode RNG seed for the kickoff + custom-spawn coin flip.
	uint32_t mutatorSeed = 0;

	// Perceiver obs layout must match TemporalObsBuilder's output for this config.
	int queryFeatures = 208;
	int kvFeatures = 69;
	int nTokens = 151;
	int nControls = 9;
	int entityFeatures = 39;
};

// Builds the C++ EnvCreateFn from an EnvConfig. The returned std::function captures config by value.
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
		// Keep holders alive for the lifetime of the env by stashing them on the match's setters;
		// they are captured by the lambda's shared_ptr copies above (kept alive via the setters).
		return { match, gym };
	};
}

// Factory that builds a Learner from an EnvConfig + LearnerConfig. This is the single generic verb
// the Python driver uses to spin up one policy's learner.
static Learner* CreateLearner(const EnvConfig& ec, LearnerConfig cfg) {
	// Keep the perceiver layout in the learner config aligned with the env config.
	cfg.ppo.perceiver.queryFeatures = ec.queryFeatures;
	cfg.ppo.perceiver.kvFeatures = ec.kvFeatures;
	cfg.ppo.perceiver.nTokens = ec.nTokens;
	cfg.ppo.perceiver.nControls = ec.nControls;
	cfg.ppo.perceiver.entityFeatures = ec.entityFeatures;
	return new Learner(MakeEnvCreateFn(ec), cfg);
}

PYBIND11_MODULE(rlgym_learner, m) {
	m.doc() = "Single-policy C++ PPO learner (EARL Perceiver) exposed to Python. "
		"League/multi-policy behavior is composed in Python by driving several of these.";

	py::enum_<LearnerDeviceType>(m, "LearnerDeviceType")
		.value("AUTO", LearnerDeviceType::AUTO)
		.value("CPU", LearnerDeviceType::CPU)
		.value("GPU_CUDA", LearnerDeviceType::GPU_CUDA)
		.export_values();

	py::enum_<HeadType>(m, "HeadType")
		.value("CONTINUOUS", HeadType::CONTINUOUS)
		.value("DISCRETE", HeadType::DISCRETE)
		.export_values();

	// Arena game modes (mirrors RocketSim::GameMode). The league builds one env per mode.
	py::enum_<GameMode>(m, "GameMode")
		.value("SOCCAR", GameMode::SOCCAR)
		.value("HOOPS", GameMode::HOOPS)
		.value("HEATSEEKER", GameMode::HEATSEEKER)
		.value("SNOWDAY", GameMode::SNOWDAY)
		.value("DROPSHOT", GameMode::DROPSHOT)
		.value("SPIKE_RUSH", GameMode::SPIKE_RUSH)
		.value("GRIDIRON", GameMode::GRIDIRON)
		.export_values();

	py::class_<PerceiverConfig>(m, "PerceiverConfig")
		.def(py::init<>())
		.def_readwrite("nDims", &PerceiverConfig::nDims)
		.def_readwrite("nHeads", &PerceiverConfig::nHeads)
		.def_readwrite("nLayers", &PerceiverConfig::nLayers)
		.def_readwrite("nControls", &PerceiverConfig::nControls)
		.def_readwrite("headType", &PerceiverConfig::headType)
		.def_readwrite("nBins", &PerceiverConfig::nBins)
		.def_readwrite("queryFeatures", &PerceiverConfig::queryFeatures)
		.def_readwrite("kvFeatures", &PerceiverConfig::kvFeatures)
		.def_readwrite("nTokens", &PerceiverConfig::nTokens)
		.def_readwrite("entityFeatures", &PerceiverConfig::entityFeatures)
		.def_readwrite("chunkSize", &PerceiverConfig::chunkSize);

	py::class_<PPOLearnerConfig>(m, "PPOLearnerConfig")
		.def(py::init<>())
		.def_readwrite("perceiver", &PPOLearnerConfig::perceiver)
		.def_readwrite("batchSize", &PPOLearnerConfig::batchSize)
		.def_readwrite("epochs", &PPOLearnerConfig::epochs)
		.def_readwrite("policyLR", &PPOLearnerConfig::policyLR)
		.def_readwrite("criticLR", &PPOLearnerConfig::criticLR)
		.def_readwrite("entCoef", &PPOLearnerConfig::entCoef)
		.def_readwrite("clipRange", &PPOLearnerConfig::clipRange)
		.def_readwrite("miniBatchSize", &PPOLearnerConfig::miniBatchSize)
		.def_readwrite("lambdaFSP", &PPOLearnerConfig::lambdaFSP)
		.def_readwrite("autocastLearn", &PPOLearnerConfig::autocastLearn)
		.def_readwrite("halfPrecModels", &PPOLearnerConfig::halfPrecModels);

	py::class_<SkillTrackerConfig>(m, "SkillTrackerConfig")
		.def(py::init<>())
		.def_readwrite("enabled", &SkillTrackerConfig::enabled);

	py::class_<LearnerConfig>(m, "LearnerConfig")
		.def(py::init<>())
		.def_readwrite("numThreads", &LearnerConfig::numThreads)
		.def_readwrite("numGamesPerThread", &LearnerConfig::numGamesPerThread)
		.def_readwrite("minInferenceSize", &LearnerConfig::minInferenceSize)
		.def_readwrite("renderMode", &LearnerConfig::renderMode)
		.def_readwrite("renderTimeScale", &LearnerConfig::renderTimeScale)
		.def_readwrite("renderDuringTraining", &LearnerConfig::renderDuringTraining)
		.def_readwrite("timestepLimit", &LearnerConfig::timestepLimit)
		.def_readwrite("expBufferSize", &LearnerConfig::expBufferSize)
		.def_readwrite("timestepsPerIteration", &LearnerConfig::timestepsPerIteration)
		.def_readwrite("standardizeReturns", &LearnerConfig::standardizeReturns)
		.def_readwrite("deterministic", &LearnerConfig::deterministic)
		.def_readwrite("collectionDuringLearn", &LearnerConfig::collectionDuringLearn)
		.def_readwrite("ppo", &LearnerConfig::ppo)
		.def_readwrite("gaeLambda", &LearnerConfig::gaeLambda)
		.def_readwrite("gaeGamma", &LearnerConfig::gaeGamma)
		.def_readwrite("rewardClipRange", &LearnerConfig::rewardClipRange)
		.def_readwrite("checkpointLoadFolder", &LearnerConfig::checkpointLoadFolder)
		.def_readwrite("checkpointSaveFolder", &LearnerConfig::checkpointSaveFolder)
		.def_readwrite("saveFolderAddUnixTimestamp", &LearnerConfig::saveFolderAddUnixTimestamp)
		.def_readwrite("timestepsPerSave", &LearnerConfig::timestepsPerSave)
		.def_readwrite("randomSeed", &LearnerConfig::randomSeed)
		.def_readwrite("checkpointsToKeep", &LearnerConfig::checkpointsToKeep)
		.def_readwrite("deviceType", &LearnerConfig::deviceType)
		.def_readwrite("sendMetrics", &LearnerConfig::sendMetrics)
		.def_readwrite("metricsProjectName", &LearnerConfig::metricsProjectName)
		.def_readwrite("metricsGroupName", &LearnerConfig::metricsGroupName)
		.def_readwrite("metricsRunName", &LearnerConfig::metricsRunName)
		.def_readwrite("skillTrackerConfig", &LearnerConfig::skillTrackerConfig);

	py::class_<EnvConfig>(m, "EnvConfig")
		.def(py::init<>())
		.def_readwrite("tickSkip", &EnvConfig::tickSkip)
		.def_readwrite("noTouchTimeoutSecs", &EnvConfig::noTouchTimeoutSecs)
		.def_readwrite("faceBallWeight", &EnvConfig::faceBallWeight)
		.def_readwrite("velPlayerToBallWeight", &EnvConfig::velPlayerToBallWeight)
		.def_readwrite("velBallToGoalWeight", &EnvConfig::velBallToGoalWeight)
		.def_readwrite("eventTeamGoal", &EnvConfig::eventTeamGoal)
		.def_readwrite("eventConcede", &EnvConfig::eventConcede)
		.def_readwrite("eventWeight", &EnvConfig::eventWeight)
		.def_readwrite("teamSize", &EnvConfig::teamSize)
		.def_readwrite("spawnOpponents", &EnvConfig::spawnOpponents)
		.def_readwrite("gameMode", &EnvConfig::gameMode)
		.def_readwrite("spawnCurriculumEnabled", &EnvConfig::spawnCurriculumEnabled)
		.def_readwrite("spawnCurriculumRatio", &EnvConfig::spawnCurriculumRatio)
		.def_readwrite("spawnCurriculumProb", &EnvConfig::spawnCurriculumProb)
		.def_readwrite("mutatorSeed", &EnvConfig::mutatorSeed)
		.def_readwrite("queryFeatures", &EnvConfig::queryFeatures)
		.def_readwrite("kvFeatures", &EnvConfig::kvFeatures)
		.def_readwrite("nTokens", &EnvConfig::nTokens)
		.def_readwrite("nControls", &EnvConfig::nControls)
		.def_readwrite("entityFeatures", &EnvConfig::entityFeatures);

	// Report is a metric map; expose it as a plain dict-like with a few conveniences.
	py::class_<Report>(m, "Report")
		.def(py::init<>())
		.def("has", &Report::Has)
		.def("get", [](const Report& r, const std::string& k) { return r.data.count(k) ? r.data.at(k) : 0.0; })
		.def("keys", [](const Report& r) { std::vector<std::string> ks; for (auto& p : r.data) ks.push_back(p.first); return ks; })
		.def("as_dict", [](const Report& r) { return r.data; })
		.def("to_string", &Report::ToString);

	// The single-policy learner handle. Methods are the generic verbs a Python driver interleaves.
	py::class_<Learner>(m, "Learner")
		.def("start_agents", &Learner::StartAgents)
		.def("stop_agents", &Learner::StopAgents)
		.def("run_iteration", &Learner::RunIteration)
		.def("learn", &Learner::Learn)
		.def("save", &Learner::Save)
		.def("load", &Learner::Load)
		.def("update_learning_rates", &Learner::UpdateLearningRates)
		.def("get_all_game_metrics", &Learner::GetAllGameMetrics)
		.def_property_readonly("obs_size", [](const Learner& l) { return l.obsSize; })
		.def_property_readonly("action_amount", [](const Learner& l) { return l.actionAmount; })
		.def_property_readonly("total_timesteps", [](const Learner& l) { return l.totalTimesteps; })
		.def_property_readonly("total_epochs", [](const Learner& l) { return l.totalEpochs; });

	m.def("create_learner", &CreateLearner,
		"Build a single-policy learner from an env config + learner config. "
		"Returns an owned handle; the Python driver may hold several to compose a league.");
}
