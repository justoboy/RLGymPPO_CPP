#pragma once
#include "../PPO/GaussianPolicy.h";
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
		torch::optim::Adam *policyOptimizer, *valueOptimizer;
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

		void UpdateLearningRates(float policyLR, float criticLR);
	};
}
