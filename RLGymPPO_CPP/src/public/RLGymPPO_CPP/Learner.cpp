#include "Learner.h"
#include "Learner.h"

#include "../../private/RLGymPPO_CPP/Util/SkillTracker.h"

#include <RLGymPPO_CPP/PPO/PPOLearner.h>
#include <RLGymPPO_CPP/PPO/ExperienceBuffer.h>
#include <RLGymPPO_CPP/Threading/ThreadAgentManager.h>

// Live-env mutator/reward types the runtime curriculum setters downcast to.
#include <RLGymSim_CPP/Utils/StateSetters/ModeSamplerMutator.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/RewardTerms.h>

#include <torch/cuda.h>
#include <ATen/Context.h>
#include "../libsrc/json/nlohmann/json.hpp"
#include <pybind11/embed.h>

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDACachingAllocator.h>
#endif

// The learner embeds CPython for the metric/render receivers. When the library is linked into a
// standalone executable we OWN the interpreter and must initialize/finalize it. When it is instead
// imported as a Python extension module (the league driver imports it), the interpreter is already
// running and owned by Python — we must NOT initialize or finalize it. We only finalize if WE were
// the ones who initialized it.
static bool s_interpreterOwnedByUs = false;

RLGPC::Learner::Learner(EnvCreateFn envCreateFn, LearnerConfig _config) :
	envCreateFn(envCreateFn),
	config(_config)
{
	if (!Py_IsInitialized()) {
		pybind11::initialize_interpreter();
		s_interpreterOwnedByUs = true;
	}

#ifndef NDEBUG
	RG_LOG("===========================");
	RG_LOG("WARNING: RLGym-PPO runs extremely slowly in debug, and there are often bizzare issues with debug-mode torch.");
	RG_LOG("It is recommended that you compile in release mode without optimization for debugging.");
	RG_SLEEP(1000);
#endif

	if (config.timestepsPerSave == 0)
		config.timestepsPerSave = config.timestepsPerIteration;

	if (config.standardizeOBS)
		RG_ERR_CLOSE("LearnerConfig.standardizeOBS has not yet been implemented, sorry");

	RG_LOG("Learner::Learner():");

	if (config.renderMode && !config.renderDuringTraining) {
		RG_LOG("\tRender mode is enabled, overriding:");
		config.numThreads = config.numGamesPerThread = 1;
		RG_LOG("\t > numThreads, numGamesPerThread = 1");

		config.sendMetrics = false;
		RG_LOG("\t > sendMetrics = false");

		config.checkpointSaveFolder.clear();
		RG_LOG("\t > checkpointSaveFolder = none");

		config.timestepsPerIteration = INT_MAX;
		RG_LOG("\t > timestepsPerIteration = inf");
	}

	if (config.saveFolderAddUnixTimestamp && !config.checkpointSaveFolder.empty())
		config.checkpointSaveFolder += "-" + std::to_string(time(0));

	RG_LOG("\tCheckpoint Load Dir: " << config.checkpointLoadFolder);
	RG_LOG("\tCheckpoint Save Dir: " << config.checkpointSaveFolder);

	torch::manual_seed(config.randomSeed);

	at::Device device = at::Device(at::kCPU);
	if (
		config.deviceType == LearnerDeviceType::GPU_CUDA || 
		(config.deviceType == LearnerDeviceType::AUTO && torch::cuda::is_available())
		) {
		RG_LOG("\tUsing CUDA GPU device...");

		// Test out moving a tensor to GPU and back to make sure the device is working
		torch::Tensor t;
		bool deviceTestFailed = false;
		try {
			t = torch::tensor(0);
			t = t.to(at::Device(at::kCUDA));
			t = t.cpu();
		} catch (...) {
			deviceTestFailed = true;
		}

		if (!torch::cuda::is_available() || deviceTestFailed)
			RG_ERR_CLOSE(
				"Learner::Learner(): Can't use CUDA GPU because " <<
				(torch::cuda::is_available() ? "libtorch cannot access the GPU" : "CUDA is not available to libtorch") << ".\n" <<
				"Make sure your libtorch comes with CUDA support, and that CUDA is installed properly."
			)
		device = at::Device(at::kCUDA);
	} else {
		RG_LOG("\tUsing CPU device...");
		device = at::Device(at::kCPU);
	}

	torch::set_num_interop_threads(1);
	torch::set_num_threads(1);

	// O3a port: enable the TF32 tensor-core matmul/conv path on the CUDA device (no-op on CPU). This is
	// a global libtorch flag, set once here at learner construction so every downstream forward/backward
	// (collection infer + PPO learn) uses it. Mirrors the reference torch.backends.cuda.matmul.allow_tf32.
	if (config.useTF32 && device.is_cuda()) {
#ifdef RG_CUDA_SUPPORT
		at::globalContext().setAllowTF32CuBLAS(true);
		at::globalContext().setAllowTF32CuDNN(true);
		RG_LOG("\tTF32 enabled for cuBLAS + cuDNN (O3a port)");
#else
		RG_LOG("\tWARNING: config.useTF32 requested but build has no CUDA support; ignoring.");
#endif
	}

	if (RocketSim::GetStage() != RocketSimStage::INITIALIZED) {
		RG_LOG("\tInitializing RocketSim...");
		RocketSim::Init(config.collisionMeshesFolder, true);
	}

	{
		RG_LOG("\tCreating test environment to determine OBS size and action amount...")
		auto envCreateResult = envCreateFn();
		auto obsSet = envCreateResult.gym->Reset();
		obsSize = obsSet[0].size();
		actionAmount = envCreateResult.match->actionParser->GetActionAmount();
		RG_LOG("\t\tOBS size: " << obsSize);
		RG_LOG("\t\tAction amount: " << actionAmount);
		delete envCreateResult.gym;
		delete envCreateResult.match;
	}

	// Resolve the per-member Perceiver specs. Empty memberPerceiver => single member (ppo.perceiver).
	std::vector<PerceiverConfig> memberSpecs;
	if (config.memberPerceiver.empty())
		memberSpecs.push_back(config.ppo.perceiver);
	else
		memberSpecs = config.memberPerceiver;

	// The superset (widest) member layout must match the actual flat obs width from the env.
	// Flat obs = q(queryFeatures) ++ kv(nTokens*kvFeatures) ++ m(nTokens) ++ mode(1).
	{
		PerceiverConfig sup = memberSpecs[0];
		for (auto& ms : memberSpecs)
			sup = (ms.entityFeatures > sup.entityFeatures) ? ms : sup;
		if (sup.queryFeatures + sup.nTokens * sup.kvFeatures + sup.nTokens + 1 != (int)obsSize)
			RG_ERR_CLOSE(
				"Learner: superset PerceiverConfig obs layout (" << sup.queryFeatures << " + "
				<< sup.nTokens << "*" << sup.kvFeatures << " + " << sup.nTokens << " + 1 = "
				<< (sup.queryFeatures + sup.nTokens * sup.kvFeatures + sup.nTokens + 1)
				<< ") does not match env obs size " << obsSize << "; fix LearnerConfig member perceiver"
			);
		if (sup.nControls != actionAmount)
			RG_ERR_CLOSE("Learner: PerceiverConfig.nControls (" << sup.nControls << ") != env action amount " << actionAmount);
	}

	// One PPO learner + experience buffer + return-stat per member (owner id).
	for (size_t g = 0; g < memberSpecs.size(); g++) {
		RG_LOG("\tCreating PPO Learner for member " << g << "...");
		PPOLearnerConfig mcfg = config.ppo;
		mcfg.perceiver = memberSpecs[g];
		ppos.push_back(new PPOLearner(mcfg, device));

		RG_LOG("\tCreating experience buffer for member " << g << "...");
		buffers.push_back(new ExperienceBuffer(config.expBufferSize, config.randomSeed + (int)g, device));

		returnStats.push_back(WelfordRunningStat(1));
	}

	// Hand the manager the per-member policies. Per-episode slot->owner routing lives in each GameInst
	// (read from GameState::slotOwners stamped by the active-set setter), not in a frozen owner map.
	// The manager's constructor derives the superset layout + per-owner column gathers from the policies
	// (single-policy when there's exactly one).
	RG_LOG("\tCreating agent manager...");
	std::vector<GaussianPolicy*> memberPolicies;
	for (auto* ppo : ppos)
		memberPolicies.push_back(ppo->policy);
	agentMgr = new ThreadAgentManager(
		memberPolicies,
		config.standardizeOBS, config.deterministic, device.is_cpu() && torch::get_num_threads() > 1,
		(uint64_t)(config.timestepsPerIteration * 1.5f),
		device
	);
	agentMgr->nControls = actionAmount;
	// Match-based collection (the reference implementation's curriculum driver): when set, each
	// iteration collects this many completed episodes instead of a raw step target (timestepsPerIteration
	// then only guards against a hung match). 0 keeps the legacy step-target behavior.
	agentMgr->matchesPerIteration = (uint64_t)config.matchesPerIteration;

	RG_LOG("\tCreating " << config.numThreads << " agents...");
	agentMgr->CreateAgents(envCreateFn, config.numThreads, config.numGamesPerThread);

	if (config.renderMode) {
		renderSender = new RenderSender();
		agentMgr->renderSender = renderSender;
		agentMgr->renderTimeScale = config.renderTimeScale;
		agentMgr->renderDuringTraining = config.renderDuringTraining;
	} else {
		renderSender = NULL;
	}

	if (config.skillTrackerConfig.enabled) {
		if (config.skillTrackerConfig.envCreateFunc == NULL)
			config.skillTrackerConfig.envCreateFunc = envCreateFn;

		skillTracker = new SkillTracker(config.skillTrackerConfig, renderSender);
	} else {
		skillTracker = NULL;
	}

	if (!config.checkpointLoadFolder.empty())
		Load();

	if (config.sendMetrics) {
		if (!runID.empty())
			RG_LOG("\tRun ID: " << runID);
		metricSender = new MetricSender(config.metricsProjectName, config.metricsGroupName, config.metricsRunName, runID);
	} else {
		metricSender = NULL;
	}
}

template <typename T>
nlohmann::json MakeJSONArray(const std::vector<T> list) {
	auto result = nlohmann::json::array();

	for (T v : list) {
		if (isnan(v))
			RG_LOG("MakeJSONArray(): Failed to serialize JSON with NAN value (list size: " << list.size() << ")");
		result.push_back(v);
	}

	return result;
}

void RLGPC::Learner::SaveStats(std::filesystem::path path, int member) {
	using namespace nlohmann;

	constexpr const char* ERROR_PREFIX = "Learner::SaveStats(): ";

	std::ofstream fOut(path);
	if (!fOut.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = {};
	j["cumulative_timesteps"] = totalTimesteps;
	j["cumulative_model_updates"] = ppos[member]->cumulativeModelUpdates;
	j["epoch"] = totalEpochs;
	
	if (skillTracker) {
		if (skillTracker->config.perModeRatings) {
			json ratings = {};
			for (auto& pair : skillTracker->curRating.data)
				ratings[pair.first] = pair.second;
			j["skill_rating"] = ratings;
		} else {
			j["skill_rating"] = skillTracker->curRating.data[""];
		}
	}

	auto& rrs = j["reward_running_stats"];
	{
		auto& rs = returnStats[member];
		rrs["mean"] = MakeJSONArray(rs.runningMean);
		rrs["var"] = MakeJSONArray(rs.runningVariance);
		rrs["shape"] = rs.shape;
		rrs["count"] = rs.count;
	}

	if (config.sendMetrics)
		j["run_id"] = metricSender->curRunID;

	std::string jStr = j.dump(4);
	fOut << jStr;
}

void RLGPC::Learner::LoadStats(std::filesystem::path path, int member) {
	// TODO: Repetitive code, merge repeated code into one function called from both SaveStats() and LoadStats()

	using namespace nlohmann;
	constexpr const char* ERROR_PREFIX = "Learner::LoadStats(): ";

	std::ifstream fIn(path);
	if (!fIn.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = json::parse(fIn);
	totalTimesteps = j["cumulative_timesteps"];
	ppos[member]->cumulativeModelUpdates = j["cumulative_model_updates"];
	totalEpochs = j["epoch"];
	
	if (skillTracker && j.contains("skill_rating")) {
		skillTracker->curRating = skillTracker->LoadRatingSet(j["skill_rating"]);
	}

	auto& rrs = j["reward_running_stats"];
	{
		WelfordRunningStat rs(rrs["shape"]);
		rs.runningMean = rrs["mean"].get<std::vector<double>>();
		rs.runningVariance = rrs["var"].get<std::vector<double>>();
		rs.count = rrs["count"];
		returnStats[member] = rs;
	}

	if (j.contains("run_id"))
		runID = j["run_id"];
}

// Different than RLGym-PPO to show that they are not compatible
constexpr const char* STATS_FILE_NAME = "RUNNING_STATS.json";

void RLGPC::Learner::Save() {
	if (config.checkpointSaveFolder.empty())
		RG_ERR_CLOSE("Learner::Save(): Cannot save because config.checkpointSaveFolder is not set");

	std::filesystem::path saveFolder = config.checkpointSaveFolder / std::to_string(totalTimesteps);
	std::filesystem::create_directories(saveFolder);

	RG_LOG("Saving to folder " << saveFolder << "...");
	for (size_t g = 0; g < ppos.size(); g++) {
		std::filesystem::path memberFolder = saveFolder / ("member_" + std::to_string(g));
		std::filesystem::create_directories(memberFolder);
		SaveStats(memberFolder / STATS_FILE_NAME, (int)g);
		ppos[g]->SaveTo(memberFolder);
	}

	// Remove old checkpoints
	if (config.checkpointsToKeep != -1) {
		int numCheckpoints = 0;
		int64_t lowestCheckpointTS = INT64_MAX;

		for (auto entry : std::filesystem::directory_iterator(config.checkpointLoadFolder)) {
			if (entry.is_directory()) {
				auto name = entry.path().filename();
				try {
					int64_t nameVal = std::stoll(name);
					lowestCheckpointTS = RS_MIN(nameVal, lowestCheckpointTS);
					numCheckpoints++;
				} catch (...) {}
			}
		}

		if (numCheckpoints > config.checkpointsToKeep) {
			std::filesystem::path removePath = config.checkpointLoadFolder / std::to_string(lowestCheckpointTS);
			try {
				std::filesystem::remove_all(removePath);
			} catch (std::exception& e) {
				RG_ERR_CLOSE("Failed to remove old checkpoint from " << removePath << ", exception: " << e.what());
			}
		}
	}

	RG_LOG(" > Done.");
}

void RLGPC::Learner::Load() {
	if (config.checkpointLoadFolder.empty())
		RG_ERR_CLOSE("Learner::Load(): Cannot load because config.checkpointLoadFolder is not set");

	RG_LOG("Loading most recent checkpoint in " << config.checkpointLoadFolder << "...");

	int64_t highest = -1;
	if (std::filesystem::is_directory(config.checkpointLoadFolder)) {
		for (auto entry : std::filesystem::directory_iterator(config.checkpointLoadFolder)) {
			if (entry.is_directory()) {
				auto name = entry.path().filename();
				try {
					int64_t nameVal = std::stoll(name);
					highest = RS_MAX(nameVal, highest);
				} catch (...) {}
			}
		}
	}

	if (highest != -1) {
		std::filesystem::path loadFolder = config.checkpointLoadFolder / std::to_string(highest);
		RG_LOG(" > Loading checkpoint " << loadFolder << "...");
		for (size_t g = 0; g < ppos.size(); g++) {
			std::filesystem::path memberFolder = loadFolder / ("member_" + std::to_string(g));
			LoadStats(memberFolder / STATS_FILE_NAME, (int)g);
			ppos[g]->LoadFrom(memberFolder);
		}
		RG_LOG(" > Done.");

		if (config.skillTrackerConfig.loadOldVersionsFromCheckpoints && ppos.size() == 1) {
			RG_LOG("Attempting to load " << config.skillTrackerConfig.maxVersions << " old versions for skill tracker...");

			int64_t targetInterval = config.skillTrackerConfig.timestepsPerVersion;
			int64_t targetTimesteps = (int64_t)totalTimesteps;

			int64_t maxAcceptableOverage = targetInterval;

			for (int i = 0; i < config.skillTrackerConfig.maxVersions; i++) {
				targetTimesteps -= targetInterval;
				
				nlohmann::json bestRating = {};
				int64_t bestTimesteps = -1;
				for (auto entry : std::filesystem::directory_iterator(config.checkpointLoadFolder)) {
					if (entry.is_directory()) {
						auto name = entry.path().filename();
						try {
							int64_t nameVal = std::stoll(name);

							if (nameVal < targetTimesteps + targetInterval) {
								if (bestTimesteps == -1 || abs(nameVal - targetTimesteps) < abs(bestTimesteps - targetTimesteps)) {

									std::ifstream fIn(entry.path() / STATS_FILE_NAME);
									if (fIn.good()) {
										using namespace nlohmann;
										json j = json::parse(fIn);

										if (j.contains("skill_rating")) {
											bestRating = j["skill_rating"];
											bestTimesteps = nameVal;
										}
									}
								}
							}

						} catch (...) {}
					}
				}

				if (bestTimesteps != -1 && bestTimesteps >= targetTimesteps - maxAcceptableOverage) {
					RG_LOG(
						" > [" << i << "]: Found at " << bestTimesteps << 
						" (target = " << targetTimesteps << ", delta = " << (targetTimesteps - bestTimesteps) << "), " <<
						"rating: " << bestRating
					);

					auto oldPolicy = ppos[0]->LoadAdditionalPolicy(config.checkpointLoadFolder / std::to_string(bestTimesteps));

					if (oldPolicy) {
						skillTracker->AppendOldPolicy(
							oldPolicy,
							skillTracker->LoadRatingSet(bestRating)
						);
					} else {
						RG_LOG(" > FAILED to load policy, policy does not exist in checkpoint!")
					}

				} else {
					RG_LOG(" > [" << i << "]: None found");
				}
			}
		}
	} else {
		RG_LOG(" > No checkpoints found, starting new model.")
	}
}

// Prints the metrics report in a similar way to rlgym-ppo
void DisplayReport(const RLGPC::Report& report) {
	// FORMAT:
	//	blank line = print blank line
	//	'-' before name = indent with dashes and spaces
	constexpr const char* REPORT_DATA_ORDER[] = {
		"Average Episode Reward",
		"Average Step Reward",
		"Policy Entropy",
		"Value Function Loss",
		"",
		"Mean KL Divergence",
		"SB3 Clip Fraction",
		"Policy Update Magnitude",
		"Value Function Update Magnitude",
		"",
		"Collected Steps/Second",
		"Overall Steps/Second",
		"",
		"Collection Time",
		"-Policy Infer Time",
		"-Env Step Time",
		"Consumption Time",
		"-PPO Learn Time",
		"Collect-Consume Overlap Time",
		// TODO: These timers don't work due to non-blocking mode
		//"--PPO Value Estimate Time",
		//"--PPO Backprop Data Time",
		//"--PPO Gradient Time",
		"Total Iteration Time",
		"",
		"Cumulative Model Updates",
		"Cumulative Timesteps",
		"",
		"Timesteps Collected"
	};

	for (const char* name : REPORT_DATA_ORDER) {
		if (strlen(name) > 0) {
			int indentLevel = 0;
			while (name[0] == '-') {
				indentLevel++;
				name++;
			}

			std::string prefix = {};
			if (indentLevel > 0) {
				prefix += std::string((indentLevel - 1) * 3, ' ');
				prefix += " - ";
			}

			RG_LOG(prefix << report.SingleToString(name, true));
		} else {
			RG_LOG("");
		}
	}
}

void RLGPC::Learner::StartAgents() {
	RG_LOG("Learner::StartAgents(): Starting agents...");
	agentMgr->SetStepCallback(stepCallback);
	agentMgr->StartAgents();
	epochTimer.Reset();
	tsSinceSave = 0;
}

void RLGPC::Learner::StopAgents() {
	RG_LOG("Learner::StopAgents(): Stopping agents...");
	agentMgr->StopAgents();
}

// One collect+learn iteration. This is the body that used to live inside Learn()'s while loop,
// lifted out so an external driver can interleave multiple single-policy learners (Jacobi league:
// freeze all -> collect -> update each). Behavior is identical to the inlined loop body.
std::vector<RLGPC::Report> RLGPC::Learner::RunIteration() {
	std::vector<Report> reports(ppos.size());

	agentMgr->SetStepCallback(stepCallback);

	// Collect the desired timesteps from our agents. One trajectory per member (owner id).
	std::vector<GameTrajectory> memberTrajs = agentMgr->CollectTimesteps(config.timestepsPerIteration);
	double relCollectionTime = epochTimer.Elapsed();
	uint64_t timestepsCollected = 0;
	for (auto& t : memberTrajs)
		timestepsCollected += t.size; // Use actual size instead of target size

	totalTimesteps += timestepsCollected;

	if (config.ppo.policyLR == 0 && config.ppo.criticLR == 0) {
		RG_LOG("\tBoth LRs are set to zero. Skipping consumption!");
#ifdef RG_CUDA_SUPPORT
		if (ppos[0]->device.is_cuda())
			c10::cuda::CUDACachingAllocator::emptyCache();
#endif
		epochTimer.Reset();
		agentMgr->ResetMetrics();
		return reports;
	}

	if (!config.collectionDuringLearn)
		agentMgr->disableCollection = true;

	// Add each member's experience to its own buffer, also computing GAE in the process
	for (size_t g = 0; g < memberTrajs.size(); g++) {
		try {
			AddNewExperience(memberTrajs[g], (int)g, reports[g]);
		} catch (std::exception& e) {
			RG_ERR_CLOSE("Exception during Learner::AddNewExperience(): " << e.what());
		}
	}

	Timer ppoLearnTimer = {};

	// Stop agents from inferencing during learn if we are not on CPU
	// This is because learning is very GPU intensive, and letting iterations collect during that time slows it down
	// On CPU, learning is its own thread, it's better to keep collecting
	// Also, if config.collectionDuringLearn is false, we ignore this
	bool blockAgentInferDuringLearn = config.collectionDuringLearn && !ppos[0]->device.is_cpu();
	{ // Run the actual PPO learning on the experience we have collected

		if (config.deterministic) {
			RG_ERR_CLOSE(
				"Learner::RunIteration(): Cannot run PPO learn iteration when on deterministic mode!"
				"\nDeterministic mode is meant for performing, not training. Only collection should occur."
			);
		}

		RG_LOG("Learning...");
		if (blockAgentInferDuringLearn)
			agentMgr->disableCollection = true;

		for (size_t g = 0; g < ppos.size(); g++) {
			try {
				ppos[g]->Learn(buffers[g], reports[g]);
			} catch (std::exception& e) {
				RG_ERR_CLOSE("Exception during PPOLearner::Learn(): " << e.what());
			}
		}

		if (blockAgentInferDuringLearn)
			agentMgr->disableCollection = false;

		totalEpochs += config.ppo.epochs;
	}

	// Free CUDA cache
#ifdef RG_CUDA_SUPPORT
	if (ppos[0]->device.is_cuda())
		c10::cuda::CUDACachingAllocator::emptyCache();
#endif

	double ppoLearnTime = ppoLearnTimer.Elapsed();
	double relEpochTime = epochTimer.Elapsed();
	epochTimer.Reset(); // Reset now otherwise we can have issues with the timer and thread input-locking

	double consumptionTime = relEpochTime - relCollectionTime;

	if (skillTracker) {
		RG_LOG("Running skill eval game(s)...");

		if (config.skillTrackerConfig.stepCallback == NULL)
			skillTracker->config.stepCallback = stepCallback;

		skillTracker->RunGames(ppos[0]->policy, timestepsCollected);
		for (auto& pair : skillTracker->curRating.data) {
			std::string metricName = RS_STR("Skill Rating" << (pair.first.empty() ? "" : " ") << pair.first);
			reports[0][metricName] = pair.second;
		}
	}

	// Get all metrics from agent manager
	Report& report = reports[0];
	agentMgr->GetMetrics(report);

	// Per-owner outcome tallies (goals/touches per episode) for the metric-gated curriculum driver.
	agentMgr->GetOwnerMetrics(reports);

	if (!config.collectionDuringLearn) {
		agentMgr->disableCollection = false;
	}

	// If we collect during consuption, don't just measure the time we waited for to collect for steps
	// Because of collection during learn, this time could be near-zero, resulting in SPS showing some crazy number
	double trueCollectionTime = config.collectionDuringLearn ? agentMgr->lastIterationTime : relCollectionTime;
	if (blockAgentInferDuringLearn)
		trueCollectionTime -= ppoLearnTime; // We couldn't have been collecting during this time
	trueCollectionTime = RS_MAX(trueCollectionTime, relCollectionTime);

	// Fix same issue with epoch time
	double trueEpochTime = RS_MAX(relEpochTime, trueCollectionTime);

	{ // Add timers to report
		report["Total Iteration Time"] = relEpochTime;

		report["Collection Time"] = relCollectionTime;
		report["Consumption Time"] = consumptionTime;
		report["Collect-Consume Overlap Time"] = (trueCollectionTime - relCollectionTime);
	}

	{ // Add timestep data to report
		report["Collected Steps/Second"] = (int64_t)(timestepsCollected / trueCollectionTime);
		report["Overall Steps/Second"] = (int64_t)(timestepsCollected / trueEpochTime);
		report["Timesteps Collected"] = timestepsCollected;
		report["Cumulative Timesteps"] = totalTimesteps;
	}

	// Call iteration callback
	if (iterationCallback) {
		RG_LOG("Calling iteration callback...");
		iterationCallback(this, reports);
	}

	{ // Print results
		constexpr const char* DIVIDER = "======================";
		RG_LOG("\n");
		RG_LOG(DIVIDER << DIVIDER);
		RG_LOG("ITERATION COMPLETED:\n");
		DisplayReport(report);
		RG_LOG(DIVIDER << DIVIDER);
		RG_LOG("\n");
	}

	// Update metric sender
	if (config.sendMetrics)
		metricSender->Send(report); // report aliases reports[0]

	// Save if needed
	tsSinceSave += timestepsCollected;
	if (tsSinceSave > config.timestepsPerSave && !config.checkpointSaveFolder.empty()) {
		Save();
		tsSinceSave = 0;
	}

	// Reset everything
	agentMgr->ResetMetrics();

	return reports;
}

void RLGPC::Learner::Learn() {
	RG_LOG("Learner::Learn():");

#ifdef RG_PARANOID_MODE
	RG_LOG("NOTE: Paranoid mode active. Additional checks will be run that may impact performance.");
#endif

	StartAgents();

	RG_LOG("\tBeginning learning loop:");
	while (totalTimesteps < config.timestepLimit || config.timestepLimit == 0)
		RunIteration();

	RG_LOG("Learner: Timestep limit of " << config.timestepLimit << " reached, stopping");
	StopAgents();
}

void RLGPC::Learner::AddNewExperience(GameTrajectory& gameTraj, int member, Report& report) {
	RG_NOGRAD;

	PPOLearner* ppo = ppos[member];
	ExperienceBuffer* expBuffer = buffers[member];

	RG_LOG("Adding experience...");

	gameTraj.RemoveCapacity();
	auto& trajData = gameTraj.data;

	size_t count = trajData.actions.size(0);
	
	// Construct input to the value function estimator that includes the final state (which an action was not taken in)
	size_t valPredCount = count + 1;

#if 0 // Does everything in one go, but can use too much VRAM
	auto valInput = 
		torch::cat({ trajData.states, torch::unsqueeze(trajData.nextStates[count - 1], 0) })
		.to(ppo->device, true);
	auto valPredsTensor = ppo->valueNet->Forward(valInput).cpu().flatten();
#else
	// Uses minibatching
	auto valPredsTensor = torch::zeros({ (int64_t)valPredCount });
	for (int i = 0; i < valPredCount; i += ppo->config.miniBatchSize) {
		int start = i;
		int end = RS_MIN(i + ppo->config.miniBatchSize, valPredCount);
		torch::Tensor statesPart = trajData.states.slice(0, start, RS_MIN(count, end));
		if (end == valPredCount) {
			auto finalNextState = torch::unsqueeze(trajData.nextStates[count - 1], 0);
			statesPart = torch::cat({ statesPart, finalNextState });
		}
		auto valPredsPart = ppo->valueNet->Forward(statesPart.to(ppo->device, true, true)).cpu().flatten();
		RG_ASSERT(valPredsPart.size(0) == (end - start));
		valPredsTensor.slice(0, start, end).copy_(valPredsPart, true);
	}
#endif

	FList valPreds = TENSOR_TO_FLIST(valPredsTensor);

	// Free CUDA cache
#ifdef RG_CUDA_SUPPORT
	if (ppo->device.is_cuda())
		c10::cuda::CUDACachingAllocator::emptyCache();
#endif
	
	float retStd = (config.standardizeReturns ? returnStats[member].GetSTD()[0] : 1);

	// Compute GAE stuff
	torch::Tensor advantages, valueTargets;
	FList returns;
	TorchFuncs::ComputeGAE(
		TENSOR_TO_FLIST(trajData.rewards),
		TENSOR_TO_FLIST(trajData.dones),
		TENSOR_TO_FLIST(trajData.truncateds),
		valPreds,
		advantages,
		valueTargets,
		returns,
		config.gaeGamma,
		config.gaeLambda,
		retStd,
		config.rewardClipRange
	);

	float avgRet = 0;
	for (float f : returns)
		avgRet += abs(f);
	avgRet /= returns.size();
	report["Avg Return"] = avgRet / retStd;

	report["Avg Advantage"] = advantages.abs().mean().item<float>();
	report["Avg Val Target"] = valueTargets.abs().mean().item<float>();

	if (config.standardizeReturns) {
		int numToIncrement = RS_MIN(config.maxReturnsPerStatsInc, returns.size());
		returnStats[member].Increment(returns, numToIncrement);
	}

	// FSP auxiliary: target = the next obs's q_self (the self row's NON-IS_* cols, the first
	// entityFeatures-5 columns of the flat next obs; the self row's IS_* one-hot is constant
	// [1,0,0,0,0] so q_self carries all the non-constant self features). mask = 1 when the step is
	// NOT an episode end (so the FSP loss skips steps with no valid next frame). When an episode ends
	// the target is masked out, so its (arbitrary) value is irrelevant. Mirrors rollout.py:
	// target = new_o[a][0][0, 0, :ef-5], mask = 0 at term/trunc else 1.
	torch::Tensor fspTarget, fspMask;
	if (ppo->config.lambdaFSP != 0) {
		int qSelf = ppo->config.perceiver.entityFeatures - 5;
		fspTarget = trajData.nextStates.slice(1, 0, qSelf);
		fspMask = (1 - trajData.dones) * (1 - trajData.truncateds);
	}

	auto expTensors = ExperienceTensors{
			trajData.states,
			trajData.actions,
			trajData.logProbs,
			trajData.rewards,

#ifdef RG_PARANOID_MODE
			trajData.debugCounters,
#endif

			trajData.nextStates,
			trajData.dones,
			trajData.truncateds,
			valueTargets,
			advantages,
			fspTarget,
			fspMask
	};
	expBuffer->SubmitExperience(
		expTensors
	);
}

void RLGPC::Learner::SeedMember(int member, std::filesystem::path folder) {
	if (member < 0 || member >= (int)ppos.size())
		RG_ERR_CLOSE("Learner::SeedMember(): member index " << member << " out of range");
	ppos[member]->SeedFrom(folder);
}

void RLGPC::Learner::SetWheelConfig(bool touchEnabled, float touchD, float touchProb,
                                   bool goalEnabled, float goalBallY, float goalProb,
                                   float drillTime, float nativeTime,
                                   bool phase1Active, float duelTime) {
	// Mutate every live game's TeamConfigMutator wheel holder. Games using a non-league setter (plain
	// RandomState self-play) have no holder and are skipped. Locks each agent's step mutex so the worker
	// doesn't read the holder mid-write.
	for (auto* agent : agentMgr->agents) {
		agent->gameStepMutex.lock();
		for (auto* game : agent->gameInsts) {
			auto* tcm = dynamic_cast<RLGSC::TeamConfigMutator*>(game->match->stateSetter);
			if (tcm && tcm->wheel) {
				tcm->wheel->touch_enabled   = touchEnabled;
				tcm->wheel->touch_d         = touchD;
				tcm->wheel->touch_prob      = touchProb;
				tcm->wheel->goal_enabled    = goalEnabled;
				tcm->wheel->goal_ball_y     = goalBallY;
				tcm->wheel->goal_prob       = goalProb;
				tcm->wheel->drill_time      = drillTime;
				tcm->wheel->native_time     = nativeTime;
				tcm->wheel->phase1_active   = phase1Active;
				tcm->wheel->duel_time       = duelTime;
			}
		}
		agent->gameStepMutex.unlock();
	}
}

void RLGPC::Learner::SetMemberReward(int member, const std::vector<std::pair<std::string, float>>& termWeights,
                                     float teamSpirit, float opponentPunish) {
	if (member < 0 || member >= (int)ppos.size())
		RG_ERR_CLOSE("Learner::SetMemberReward(): member index " << member << " out of range");

	// Rebuild this member's reward spec on every live ComposedReward. Games using a non-composed reward
	// (plain CombinedReward self-play) are skipped. Locks each agent's step mutex for atomic mutation.
	for (auto* agent : agentMgr->agents) {
		agent->gameStepMutex.lock();
		for (auto* game : agent->gameInsts) {
			auto* cr = dynamic_cast<RLGSC::ComposedReward*>(game->match->rewardFn);
			if (!cr)
				continue;
			if (member >= (int)cr->members.size())
				cr->members.resize(member + 1);
			RLGSC::MemberRewardSpec spec;
			for (const auto& kv : termWeights)
				spec.termWeights[kv.first] = kv.second;
			spec.teamSpirit = teamSpirit;
			spec.opponentPunish = opponentPunish;
			cr->members[member] = std::move(spec);
		}
		agent->gameStepMutex.unlock();
	}
}

void RLGPC::Learner::UpdateLearningRates(float policyLR, float criticLR) {
	for (auto* p : ppos)
		p->UpdateLearningRates(policyLR, criticLR);
}

std::vector<RLGPC::Report> RLGPC::Learner::GetAllGameMetrics() {
	std::vector<Report> reports = {};

	for (auto agent : agentMgr->agents) {
		agent->gameStepMutex.lock();
		for (auto game : agent->gameInsts)
			if (!game->_metrics.data.empty())
				reports.push_back(game->_metrics);
		agent->gameStepMutex.unlock();
	}

	return reports;
}

RLGPC::Learner::~Learner() {
	for (auto* p : ppos) delete p;
	for (auto* b : buffers) delete b;
	delete agentMgr;
	delete metricSender;
	delete renderSender;
	if (s_interpreterOwnedByUs) {
		pybind11::finalize_interpreter();
		s_interpreterOwnedByUs = false;
	}
}