#include "PPOLearner.h"

#include "../Util/TorchFuncs.h"
#include "../Util/StateDictIO.h"

#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/api/include/torch/serialize.h>

using namespace torch;

Tensor _CopyParams(nn::Module& mod) {
	return torch::nn::utils::parameters_to_vector(mod.parameters()).cpu();
}

void _CopyModelParamsHalf(nn::Module* from, nn::Module* to) {
	RG_NOGRAD;
	try {
		auto fromParams = from->parameters();
		auto toParams = to->parameters();
		for (int i = 0; i < fromParams.size(); i++) {
			auto scaledParams = fromParams[i].to(RG_HALFPERC_TYPE);
			toParams[i].copy_(scaledParams, true);
		}
	} catch (std::exception& e) {
		RG_ERR_CLOSE("_CopyModelParamsHalf() exception: " << e.what());
	}
}

RLGPC::PPOLearner::PPOLearner(PPOLearnerConfig _config, Device _device) 
	: config(_config), device(_device), nControls(_config.perceiver.nControls) {

	if (config.miniBatchSize == 0)
		config.miniBatchSize = config.batchSize;

	if (config.batchSize % config.miniBatchSize != 0)
		RG_ERR_CLOSE("PPOLearner: config.batchSize must be a multiple of config.miniBatchSize");

	// Actor: EARL Perceiver + continuous Gaussian head
	policy = new GaussianPolicy(config.perceiver, device);

	// Critic: EARL Perceiver + Linear(256->1) head
	EARLPerceiver criticEarl = EARLPerceiver(
		config.perceiver.nDims, config.perceiver.nHeads, config.perceiver.nLayers,
		config.perceiver.queryFeatures, config.perceiver.kvFeatures);
	valueNet = PerceiverNet(
		criticEarl,
		torch::nn::Linear(torch::nn::LinearOptions(config.perceiver.nDims, 1)),
		config.perceiver.queryFeatures, config.perceiver.kvFeatures,
		config.perceiver.nTokens, config.perceiver.chunkSize);
	valueNet->to(device, true);

	policyOptimizer = new optim::Adam(policy->net->parameters(), optim::AdamOptions(config.policyLR));
	valueOptimizer = new optim::Adam(valueNet->parameters(), optim::AdamOptions(config.criticLR));
	valueLossFn = nn::MSELoss();

	// FSP auxiliary head: shares the actor's EARLPerceiver trunk instance (policy->net->earl) so the
	// auxiliary gradient trains the same encoder the actor uses. Only the head's own Linear params get
	// the dedicated fspOptimizer; the shared trunk stays in policyOptimizer (mirrors agent.py:346-350).
	// When lambdaFSP == 0 the head is never built and fspOptimizer is a no-op (single-bot BC->RL stays
	// byte-identical to the pre-FSP learner).
	if (config.lambdaFSP != 0) {
		fsp = FSPNet(
			policy->net->earl, config.perceiver.entityFeatures,
			config.perceiver.queryFeatures, config.perceiver.kvFeatures,
			config.perceiver.nTokens, config.perceiver.chunkSize);
		fsp->to(device, true);
		fspOptimizer = new optim::Adam(fsp->head->parameters(), optim::AdamOptions(config.policyLR));
	} else {
		fspOptimizer = NULL;
	}

	if (config.measureGradientNoise) {
		noiseTrackerPolicy = new GradNoiseTracker(config.batchSize, config.gradientNoiseUpdateInterval, config.gradientNoiseAvgDecay);
		noiseTrackerValueNet = new GradNoiseTracker(config.batchSize, config.gradientNoiseUpdateInterval, config.gradientNoiseAvgDecay);
	} else {
		noiseTrackerPolicy = NULL;
		noiseTrackerValueNet = NULL;
	}
}

void RLGPC::PPOLearner::Learn(ExperienceBuffer* expBuffer, Report& report) {
	
	bool autocast = config.autocastLearn;

	if (autocast) {
#ifndef RG_CUDA_SUPPORT
		RG_ERR_CLOSE("Autocast not supported on non-CUDA!")
#endif
	}

	static amp::GradScaler* gradScaler = NULL;
#ifdef RG_CUDA_SUPPORT
	if (autocast && !gradScaler) {
		RG_LOG("Creating grad scaler...");
		gradScaler = new amp::GradScaler();
	}
#endif

	int
		numIterations = 0,
		numMinibatchIterations = 0;
	float
		meanEntropy = 0,
		meanDivergence = 0,
		meanValLoss = 0,
		meanRatio = 0,
		meanFSP = 0;
	FList clipFractions = {};

	// Save parameters first
	auto policyBefore = _CopyParams(*policy->net);
	auto criticBefore = _CopyParams(*valueNet);

	bool trainPolicy = config.policyLR != 0;
	bool trainCritic = config.criticLR != 0;
	bool trainFSP = fsp && config.lambdaFSP != 0;

	Timer totalTimer = {};
	for (int epoch = 0; epoch < config.epochs; epoch++) {

		// Get randomly-ordered timesteps for PPO
		auto batches = expBuffer->GetAllBatchesShuffled(config.batchSize);

		for (auto& batch : batches) {
			auto batchActs = batch.actions;
			auto batchOldProbs = batch.logProbs;
			auto batchObs = batch.states;
			auto batchTargetValues = batch.values;
			auto batchAdvantages = batch.advantages;
			auto batchFspTarget = batch.fspTarget;
			auto batchFspMask = batch.fspMask;

			batchActs = batchActs.view({ config.batchSize, -1 });
			policyOptimizer->zero_grad();
			valueOptimizer->zero_grad();
			if (trainFSP)
				fspOptimizer->zero_grad();

			// https://stackoverflow.com/questions/30297465/wait-for-all-threads-in-c
			std::mutex threadLockMutex;
			std::mutex threadUpdateMutex;
			std::condition_variable threadCV;
			std::atomic<int> threadCounter = 0;

			auto fnRunMinibatch = [&](int start, int stop) {

				float batchSizeRatio = (stop - start) / (float)config.batchSize;

				// Send everything to the device and enforce correct shapes
				auto acts = batchActs.slice(0, start, stop).to(device, true, true);
				auto obs = batchObs.slice(0, start, stop).to(device, true, true);
				
				auto advantages = batchAdvantages.slice(0, start, stop).to(device, true, true);
				auto oldProbs = batchOldProbs.slice(0, start, stop).to(device, true, true);
				auto targetValues = batchTargetValues.slice(0, start, stop).to(device, true, true);

				torch::Tensor fspTgt, fspMsk;
				if (trainFSP) {
					fspTgt = batchFspTarget.slice(0, start, stop).to(device, true, true);
					fspMsk = batchFspMask.slice(0, start, stop).to(device, true, true);
				}

				Timer timer = {};
				if (autocast) RG_AUTOCAST_ON();
				auto vals = valueNet->Forward(obs); // 11%
				threadUpdateMutex.lock();
				report.Accum("PPO Value Estimate Time", timer.Elapsed());
				threadUpdateMutex.unlock();

				timer.Reset();
				torch::Tensor logProbs, entropy, ratio, clipped, policyLoss, ppoLoss;
				if (trainPolicy) {
					// Get policy log probs & entropy (Gaussian policy over the 9 control channels)
					GaussianPolicy::BackpropResult bpResult = policy->GetBackpropData(obs, acts); // 13%

					logProbs = bpResult.actionLogProbs;
					entropy = bpResult.entropy;

					logProbs = logProbs.view_as(oldProbs);
					threadUpdateMutex.lock();
					report.Accum("PPO Backprop Data Time", timer.Elapsed());
					threadUpdateMutex.unlock();

					// Compute PPO loss
					ratio = exp(logProbs - oldProbs);
					threadUpdateMutex.lock();
					meanRatio += ratio.mean().detach().cpu().item<float>();
					threadUpdateMutex.unlock();
					clipped = clamp(
						ratio, 1 - config.clipRange, 1 + config.clipRange
					);

					// Compute policy loss
					policyLoss = -min(
						ratio * advantages, clipped * advantages
					).mean();
					ppoLoss = (policyLoss - entropy * config.entCoef) * batchSizeRatio;
				}

				torch::Tensor valueLoss;
				if (trainCritic) {
					// Compute value loss
					vals = vals.view_as(targetValues);
					valueLoss = valueLossFn(vals, targetValues) * batchSizeRatio;
				}

				// FSP auxiliary loss (mirrors fsp.py:224-226): per-sample smooth_l1 over the feature
				// dim (mean), masked sum over the batch / mask.sum().clamp(min=1). fspRaw is the raw
				// (unscaled) loss reported as "FSP Loss" (comparable to Python's ppo/fsp_loss); the
				// gradient term additionally scales by lambdaFSP * batchSizeRatio (the C++ analogue of
				// Python's lambda_fsp * fsp_loss / n_minibatches).
				torch::Tensor fspRaw, fspLoss;
				if (trainFSP) {
					auto fspPred = fsp->Forward(obs); // (mb, entityFeatures)
					auto perSample = torch::nn::functional::smooth_l1_loss(
						fspPred, fspTgt,
						torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kNone)
					).mean(-1); // (mb,)
					fspRaw = (perSample * fspMsk).sum() / fspMsk.sum().clamp(1.0);
					fspLoss = fspRaw * config.lambdaFSP * batchSizeRatio;
				}

				if (autocast) RG_AUTOCAST_OFF();

				float kl;
				if (trainPolicy) {
					// Compute KL divergence & clip fraction using SB3 method for reporting
					float clipFraction;
					{
						RG_NOGRAD;

						auto logRatio = logProbs - oldProbs;
						auto klTensor = (exp(logRatio) - 1) - logRatio;
						kl = klTensor.mean().detach().cpu().item<float>();

						clipFraction = mean((abs(ratio - 1) > config.clipRange).to(kFloat)).cpu().item<float>();
						threadUpdateMutex.lock();
						clipFractions.push_back(clipFraction);
						threadUpdateMutex.unlock();
					}
				}

				//timer.Reset();
				// NOTE: These gradient calls are a substantial portion of learn time
				//	From my testing, they are around 61% of learn time
				//	Results will probably vary heavily depending on model size and GPU strength
				if (autocast) {
					if (trainPolicy)
						gradScaler->scale(ppoLoss).backward();
					if (trainCritic)
						gradScaler->scale(valueLoss).backward();
					if (trainFSP)
						gradScaler->scale(fspLoss).backward();
				} else {
					if (trainPolicy)
						ppoLoss.backward(); // 29%
					if (trainCritic)
						valueLoss.backward(); // 24%
					if (trainFSP)
						fspLoss.backward();
				}

				threadUpdateMutex.lock();
				{
					report.Accum("PPO Gradient Time", timer.Elapsed());

					if (trainCritic)
						meanValLoss += valueLoss.cpu().detach().item<float>();
					if (trainPolicy) {
						meanDivergence += kl;
						meanEntropy += entropy.cpu().detach().item<float>();
					}
					if (trainFSP)
						meanFSP += fspRaw.cpu().detach().item<float>();
					numMinibatchIterations += 1;
				}
				threadUpdateMutex.unlock();

				std::lock_guard<std::mutex> lk(threadLockMutex);
				threadCounter--;
				threadCV.notify_all();
			};

			if (this->device.is_cpu()) {

				if (!this->minibatchThreadPool) {
					int numThreads = std::thread::hardware_concurrency();
					numThreads += numThreads / 2; // Seems to be slightly faster
					this->minibatchThreadPool = new ThreadPool(numThreads);
				}

				// Use multithreaded PPO learn
				int realMinibatchSize = config.batchSize / this->minibatchThreadPool->threads.size();

				for (int mbs = 0; mbs < config.batchSize; mbs += realMinibatchSize) {
					int start = mbs;
					int stop = start + realMinibatchSize;
					stop = RS_MIN(stop, config.batchSize);

					this->minibatchThreadPool->StartJob(std::bind(fnRunMinibatch, start, stop));
				}

				while (this->minibatchThreadPool->GetNumRunningJobs() > 0)
					RG_SLEEP(1);

			} else {
				for (int mbs = 0; mbs < config.batchSize; mbs += config.miniBatchSize) {
					int start = mbs;
					int stop = start + config.miniBatchSize;
					fnRunMinibatch(start, stop);
				}
			}

			if (config.measureGradientNoise) {
				if (trainPolicy)
					noiseTrackerPolicy->Update(*policy->net);
				if (trainCritic)
					noiseTrackerValueNet->Update(*valueNet);
			}

			if (trainPolicy)
				nn::utils::clip_grad_norm_(policy->net->parameters(), 0.5f);
			if (trainCritic)
				nn::utils::clip_grad_norm_(valueNet->parameters(), 0.5f);
			if (trainFSP)
				nn::utils::clip_grad_norm_(fsp->head->parameters(), 0.5f);
			

			if (autocast) {
				if (trainPolicy)
					gradScaler->step(*policyOptimizer);
				if (trainCritic)
					gradScaler->step(*valueOptimizer);
				if (trainFSP)
					gradScaler->step(*fspOptimizer);
			} else {
				if (trainPolicy)
					policyOptimizer->step();
				if (trainCritic)
					valueOptimizer->step();
				if (trainFSP)
					fspOptimizer->step();
			}

			if (autocast)
				gradScaler->update();
			numIterations += 1;
		}
	}

	numIterations = RS_MAX(numIterations, 1);
	numMinibatchIterations = RS_MAX(numMinibatchIterations, 1);

	// Compute averages for the metrics that will be reported
	meanEntropy /= numMinibatchIterations;
	meanDivergence /= numMinibatchIterations;
	meanValLoss /= numMinibatchIterations;
	meanRatio /= numMinibatchIterations;
	meanFSP /= numMinibatchIterations;

	float meanClip = 0;
	if (!clipFractions.empty()) {
		for (float f : clipFractions)
			meanClip += f;
		meanClip /= clipFractions.size();
	}

	// Compute magnitude of updates made to the policy and value estimator
	auto policyAfter = _CopyParams(*policy->net);
	auto criticAfter = _CopyParams(*valueNet);

	float policyUpdateMagnitude = (policyBefore - policyAfter).norm().item<float>();
	float criticUpdateMagnitude = (criticBefore - criticAfter).norm().item<float>();

	float totalTime = totalTimer.Elapsed();

	// Assemble and return report
	cumulativeModelUpdates += numIterations;
	report["PPO Batch Consumption Time"] = totalTime / numIterations;
	report["Cumulative Model Updates"] = cumulativeModelUpdates;
	report["Policy Entropy"] = meanEntropy;
	report["Mean KL Divergence"] = meanDivergence;
	report["Mean Ratio"] = meanRatio;
	report["Value Function Loss"] = meanValLoss;
	if (trainFSP)
		report["FSP Loss"] = meanFSP;
	report["SB3 Clip Fraction"] = meanClip;
	report["Policy Update Magnitude"] = policyUpdateMagnitude;
	report["Value Function Update Magnitude"] = criticUpdateMagnitude;
	report["PPO Learn Time"] = totalTimer.Elapsed();

	if (config.measureGradientNoise) {
		if (noiseTrackerPolicy->lastNoiseScale != 0)
			report["Grad Noise Policy"] = noiseTrackerPolicy->lastNoiseScale;
		if (noiseTrackerValueNet->lastNoiseScale != 0)
			report["Grad Noise Value Net"] = noiseTrackerValueNet->lastNoiseScale;
	}

	policyOptimizer->zero_grad();
	valueOptimizer->zero_grad();
	if (fspOptimizer)
		fspOptimizer->zero_grad();
}

// Get sizes of all parameters in a module
std::vector<uint64_t> GetSeqSizes(torch::nn::Module& module) {
	std::vector<uint64_t> result = {};

	for (auto& param : module.parameters())
		result.push_back(param.numel());

	return result;
}

constexpr const char* MODEL_FILE_NAMES[] = {
		"PPO_POLICY.lt",
		"PPO_CRITIC.lt",
		"PPO_FSP.lt",
};

constexpr const char* OPTIM_FILE_NAMES[] = {
	"PPO_POLICY_OPTIM.lt",
	"PPO_CRITIC_OPTIM.lt",
	"PPO_FSP_OPTIM.lt",
};

void TorchLoadSaveSeq(std::shared_ptr<torch::nn::Module> module, std::filesystem::path path, c10::Device device, bool load) {
	if (load) {
		auto streamIn = std::ifstream(path, std::ios::binary);
		streamIn >> std::noskipws;

		if (!streamIn.good())
			RG_ERR_CLOSE("Failed to load from " << path << ", file does not exist or can't be accessed");

		auto sizesBefore = GetSeqSizes(*module);

		try {
			torch::load(module, streamIn, device);
		} catch (std::exception& e) {
			RG_ERR_CLOSE(
				"Failed to load model, checkpoint may be corrupt or of different model arch.\n" <<
				"Exception: " << e.what()
			);
		}

		// Torch will happily load in a model of a totally different size, then we will crash when we try to use it
		// So we need to manually check if it is the same size
		auto sizesAfter = GetSeqSizes(*module);
		if (!std::equal(sizesBefore.begin(), sizesBefore.end(), sizesAfter.begin(), sizesAfter.end())) {
			std::stringstream stream;
			stream << "Saved model has different size than current model, cannot load model from " << path << ":\n";
			
			for (int i = 0; i < 2; i++) {
				stream << " > " << (i ? "Saved model:   [ " : "Current model: [ ");
				for (uint64_t size : (i ? sizesAfter : sizesBefore))
					stream << size << ' ';

				stream << " ]";
				if (i == 0)
					stream << ",\n";
			}

			RG_ERR_CLOSE(stream.str());
		}

	} else {
		auto streamOut = std::ofstream(path, std::ios::binary);
		torch::save(module, streamOut);
	}
}

void TorchLoadSaveAll(RLGPC::PPOLearner* learner, std::filesystem::path folderPath, bool load) {

	if (load) {
		if (!std::filesystem::exists(folderPath / MODEL_FILE_NAMES[0]))
			RG_ERR_CLOSE("PPOLearner: Failed to find file \"" << MODEL_FILE_NAMES[0] << "\" in " << folderPath << ".")
	}

	TorchLoadSaveSeq(learner->policy->net.ptr(), folderPath / MODEL_FILE_NAMES[0], learner->device, load);

	if (!load || std::filesystem::exists(folderPath / MODEL_FILE_NAMES[1]))
		TorchLoadSaveSeq(learner->valueNet.ptr(), folderPath / MODEL_FILE_NAMES[1], learner->device, load);

	// FSP module (shared earl.* trunk + head.*). Saved/loaded only when the FSP head is active.
	bool hasFSP = learner->fspOptimizer != NULL;
	if (hasFSP)
		TorchLoadSaveSeq(learner->fsp.ptr(), folderPath / MODEL_FILE_NAMES[2], learner->device, load);

	// Load or save optimizers. Index 2 (FSP head) only when the FSP head is active.
	int nOptims = hasFSP ? 3 : 2;
	if (load) {
		try {
			for (int i = 0; i < nOptims; i++) {
				auto path = folderPath / OPTIM_FILE_NAMES[i];

				if (!std::filesystem::exists(path)) {
					RG_LOG("WARNING: No optimizer found at " << path << ", optimizer will be reset");
					continue;
				}

				{ // Check if empty
					std::ifstream testStream = std::ifstream(path, std::istream::ate | std::ios::binary);
					if (testStream.tellg() == 0) {
						RG_LOG("WARNING: Saved optimizer is empty, optimizer will be reset");
						continue;
					}
				}

				torch::optim::Adam* optim =
					i == 0 ? learner->policyOptimizer :
					i == 1 ? learner->valueOptimizer : learner->fspOptimizer;

				torch::serialize::InputArchive policyOptArchive;
				policyOptArchive.load_from(path.string(), learner->device);
				optim->load(policyOptArchive);
			}

		} catch (std::exception& e) {
			RG_ERR_CLOSE(
				"Failed to load optimizers, exception: " << e.what() << "\n" <<
				"Checkpoint may be corrupt."
			);
		}
	} else {
		for (int i = 0; i < nOptims; i++) {
			torch::optim::Adam* optim =
				i == 0 ? learner->policyOptimizer :
				i == 1 ? learner->valueOptimizer : learner->fspOptimizer;

			torch::serialize::OutputArchive policyOptArchive;
			optim->save(policyOptArchive);
			policyOptArchive.save_to((folderPath / OPTIM_FILE_NAMES[i]).string());
		}
	}
}

void RLGPC::PPOLearner::SaveTo(std::filesystem::path folderPath) {
	RG_LOG("PPOLearner(): Saving models to: " << folderPath);
	TorchLoadSaveAll(this, folderPath, false);
}

RLGPC::GaussianPolicy* RLGPC::PPOLearner::LoadAdditionalPolicy(std::filesystem::path folderPath) {
	std::filesystem::path policyPath = folderPath / MODEL_FILE_NAMES[0];
	if (!std::filesystem::exists(policyPath))
		return NULL;

	GaussianPolicy* newPolicy = new GaussianPolicy(config.perceiver, policy->device);
	TorchLoadSaveSeq(newPolicy->net.ptr(), policyPath, newPolicy->device, true);
	return newPolicy;
}

void RLGPC::PPOLearner::LoadFrom(std::filesystem::path folderPath)  {
	RG_LOG("PPOLearner(): Loading models from: " << folderPath);
	if (!std::filesystem::is_directory(folderPath))
		RG_ERR_CLOSE("PPOLearner:LoadFrom(): Path " << folderPath << " is not a valid directory");

	TorchLoadSaveAll(this, folderPath, true);

	UpdateLearningRates(config.policyLR, config.criticLR);
}

// WEIG seed path (cross-version). Reads the binary state-dict blobs written by
// tools/convert_bc_checkpoint.py and copies each tensor into the matching module by qualified
// name. The actor blob (PPO_POLICY.lt) is required; critic/FSP blobs are optional (loaded only when
// present + the FSP head is active). Optimizers are NOT seeded (fresh Adam state) — correct for a
// BC->RL warm start. Mirrors parity_main.cpp's load path, which is the parity-gated reference.
void RLGPC::PPOLearner::SeedFrom(std::filesystem::path folderPath) {
	RG_LOG("PPOLearner(): Seeding (WEIG) from: " << folderPath);
	if (!std::filesystem::is_directory(folderPath))
		RG_ERR_CLOSE("PPOLearner::SeedFrom(): Path " << folderPath << " is not a valid directory");

	std::filesystem::path actorPath = folderPath / MODEL_FILE_NAMES[0];
	if (!std::filesystem::exists(actorPath))
		RG_ERR_CLOSE("PPOLearner::SeedFrom(): missing actor blob " << actorPath);
	ApplyStateDict(*policy->net.ptr(), LoadStateDict(actorPath.string()));
	RG_LOG(" > Seeded actor from " << actorPath.filename().string());

	std::filesystem::path criticPath = folderPath / MODEL_FILE_NAMES[1];
	if (std::filesystem::exists(criticPath)) {
		ApplyStateDict(*valueNet.ptr(), LoadStateDict(criticPath.string()));
		RG_LOG(" > Seeded critic from " << criticPath.filename().string());
	}

	if (fspOptimizer != NULL) {
		std::filesystem::path fspPath = folderPath / MODEL_FILE_NAMES[2];
		if (std::filesystem::exists(fspPath)) {
			// FSP blob carries earl.* (same shared trunk values as the actor) + head.*; applying it
			// re-writes the shared earl with identical values and fills the head.
			ApplyStateDict(*fsp.ptr(), LoadStateDict(fspPath.string()));
			RG_LOG(" > Seeded FSP from " << fspPath.filename().string());
		}
	}
}

void RLGPC::PPOLearner::UpdateLearningRates(float policyLR, float criticLR) {
	config.policyLR = policyLR;
	config.criticLR = criticLR;

	for (auto& g : policyOptimizer->param_groups())
		static_cast<torch::optim::AdamOptions&>(g.options()).lr(policyLR);

	for (auto& g : valueOptimizer->param_groups())
		static_cast<torch::optim::AdamOptions&>(g.options()).lr(criticLR);

	std::stringstream updatedMsg;
	updatedMsg << std::scientific << "Updated learning rate to [" << policyLR << ", " << criticLR << "]";
	RG_LOG("PPOLearner: " << updatedMsg.str());
}