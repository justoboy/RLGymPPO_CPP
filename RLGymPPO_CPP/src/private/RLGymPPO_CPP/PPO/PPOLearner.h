#pragma once
#include "../PPO/GaussianPolicy.h";
#include "../Model/FSPNet.h";
#include "ExperienceBuffer.h";
#include <RLGymPPO_CPP/Util/Report.h>
#include <RLGymPPO_CPP/Util/Timer.h>
#include <RLGymPPO_CPP/PPO/PPOLearnerConfig.h>

#include <torch/optim/adam.h>
#include <torch/nn/modules/loss.h>
#include "../Util/gradscaler.hpp"
#include "../Util/GradNoiseTracker.h"
#include "../Util/ThreadPool.h"

namespace RLGPC {
	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
	// Actor is a GaussianPolicy (PerceiverNet + ContinuousControlsHead); the critic is a
	// PerceiverNet with a Linear(256->1) head. Both consume the FLAT obs and rebuild the
	// (q, kv, mask) tuple internally (chunked forward bounds VRAM).
	class PPOLearner {
	public:
		GaussianPolicy* policy;
		PerceiverNet valueNet{nullptr};

		// FSP auxiliary head (optional; active when config.lambdaFSP != 0). It holds the SAME
		// EARLPerceiver trunk instance the actor uses (policy->net->earl), so the auxiliary
		// gradient flows into the shared encoder. Only the head's own Linear params get the
		// dedicated fspOptimizer; the shared trunk stays in policyOptimizer (mirrors agent.py).
		FSPNet fsp{nullptr};

		torch::optim::Adam *policyOptimizer, *valueOptimizer, *fspOptimizer;
		torch::nn::MSELoss valueLossFn;

		GradNoiseTracker* noiseTrackerPolicy, *noiseTrackerValueNet;

		PPOLearnerConfig config;
		torch::Device device;

		int nControls;

		ThreadPool* minibatchThreadPool = NULL;

		int cumulativeModelUpdates = 0;

		PPOLearner(
			PPOLearnerConfig config, torch::Device device
		);

		void Learn(ExperienceBuffer* expBuffer, Report& report);

		void SaveTo(std::filesystem::path folderPath);
		void LoadFrom(std::filesystem::path folderPath);
		GaussianPolicy* LoadAdditionalPolicy(std::filesystem::path folderPath);

		// WEIG seed path: read the version-independent binary state-dict blobs written by
		// tools/convert_bc_checkpoint.py (PPO_POLICY.lt / PPO_CRITIC.lt / PPO_FSP.lt) and copy each
		// tensor into the matching module by qualified name. Unlike LoadFrom (torch pickle archive,
		// only readable by the SAME libtorch that wrote it), this is cross-version: a BC checkpoint
		// trained under torch 2.14 loads byte-identically into the libtorch 2.2 learner. Optimizers
		// are NOT seeded (fresh Adam state), which is correct for a BC->RL warm start.
		void SeedFrom(std::filesystem::path folderPath);

		void UpdateLearningRates(float policyLR, float criticLR);
	};
}
