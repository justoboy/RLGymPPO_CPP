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
#include <RLGymPPO_CPP/EnvFactory.h>

namespace py = pybind11;
using namespace RLGPC;
using namespace RLGSC;

// NOTE: EnvConfig + CreateLearner are DECLARED in <RLGymPPO_CPP/EnvFactory.h> and DEFINED inside the
// DLL (EnvFactory.cpp). The factory lambda that constructs the Gym/Match MUST be compiled into the
// DLL, because RLGymSim_CPP is statically linked into BOTH this module and the DLL; if the lambda
// lived here it would call `new Gym` against this module's *uninitialized* copy of RocketSim's
// static stage state (RocketSim::Init only initialized the DLL's copy) and crash. Binding the
// DLL-owned CreateLearner keeps all RLGymSim_CPP construction on the initialized (DLL) side.

PYBIND11_MODULE(rlgym_learner, m) {
	m.doc() = "C++ PPO learner (EARL Perceiver) exposed to Python. A single handle may own one or "
		"more members (policies); run_iteration() returns one metric dict per member. Multi-member "
		"behavior is composed in Python by configuring memberPerceiver/slotOwners.";

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
		// Checkpoint folders are std::filesystem::path internally; expose as str for Python ergonomics.
		.def_property("checkpointLoadFolder",
			[](const LearnerConfig& c) { return c.checkpointLoadFolder.string(); },
			[](LearnerConfig& c, const std::string& s) { c.checkpointLoadFolder = s.empty() ? std::filesystem::path{} : std::filesystem::path(s); })
		.def_property("checkpointSaveFolder",
			[](const LearnerConfig& c) { return c.checkpointSaveFolder.string(); },
			[](LearnerConfig& c, const std::string& s) { c.checkpointSaveFolder = s.empty() ? std::filesystem::path{} : std::filesystem::path(s); })
		.def_readwrite("saveFolderAddUnixTimestamp", &LearnerConfig::saveFolderAddUnixTimestamp)
		.def_readwrite("timestepsPerSave", &LearnerConfig::timestepsPerSave)
		.def_readwrite("deviceType", &LearnerConfig::deviceType)
		.def_readwrite("randomSeed", &LearnerConfig::randomSeed)
		.def_readwrite("checkpointsToKeep", &LearnerConfig::checkpointsToKeep)
		// Collision-mesh folder (absolute path recommended when embedded in a host with unknown CWD).
		.def_property("collisionMeshesFolder",
			[](const LearnerConfig& c) { return c.collisionMeshesFolder.string(); },
			[](LearnerConfig& c, const std::string& s) { c.collisionMeshesFolder = s; })
		.def_readwrite("sendMetrics", &LearnerConfig::sendMetrics)
		.def_readwrite("metricsProjectName", &LearnerConfig::metricsProjectName)
		.def_readwrite("metricsGroupName", &LearnerConfig::metricsGroupName)
		.def_readwrite("metricsRunName", &LearnerConfig::metricsRunName)
		.def_readwrite("skillTrackerConfig", &LearnerConfig::skillTrackerConfig)
		// Multi-member league config. memberPerceiver: one PerceiverConfig per member (owner id);
		// empty => single member. slotOwners: per-slot owner id (same for every game); empty => all 0.
		.def_readwrite("memberPerceiver", &LearnerConfig::memberPerceiver)
		.def_readwrite("slotOwners", &LearnerConfig::slotOwners);

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
		.def("seed_member", [](Learner& l, int member, const std::string& folder) {
			l.SeedMember(member, std::filesystem::path(folder));
		}, py::arg("member"), py::arg("folder"),
		"Seed a member's weights from a WEIG checkpoint folder (PPO_POLICY.lt etc.). Optimizers stay fresh.")
		.def("get_all_game_metrics", &Learner::GetAllGameMetrics)
		.def_property_readonly("obs_size", [](const Learner& l) { return l.obsSize; })
		.def_property_readonly("action_amount", [](const Learner& l) { return l.actionAmount; })
		.def_property_readonly("total_timesteps", [](const Learner& l) { return l.totalTimesteps; })
		.def_property_readonly("total_epochs", [](const Learner& l) { return l.totalEpochs; });

	m.def("create_learner", &CreateLearner,
		"Build a single-policy learner from an env config + learner config. "
		"Returns an owned handle; the Python driver may hold several to compose a league.");
}
