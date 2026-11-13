#pragma once
#include "../Model/PerceiverNet.h"
#include <RLGymPPO_CPP/PPO/PPOLearnerConfig.h>

// Continuous Gaussian actor wrapper (replaces the old discrete DiscretePolicy).
// Mirrors reference.agent.ContinuousPolicy: the actor PerceiverNet head outputs
// [mean(nControls) || log_std(nControls)]; policy = Independent Normal.
//   log_std.clamp(-5, 2); std = exp(log_std).clamp(1e-4, 10)
//   log_prob/entropy SUM over the control channels (independent, 1-dim event).
namespace RLGPC {
	class GaussianPolicy {
	public:
		PerceiverNet net{nullptr};
		PerceiverConfig config;
		torch::Device device;
		int nControls;

		GaussianPolicy(const PerceiverConfig& config, torch::Device device);

		RG_NO_COPY(GaussianPolicy);

		// Deep copy of this policy into a fresh GaussianPolicy (same layout config)
		GaussianPolicy* Clone();

		void CopyTo(GaussianPolicy& to) { net->CopyTo(to.net); }

		struct ActionResult {
			torch::Tensor action, logProb;
		};
		ActionResult GetAction(torch::Tensor obs, bool deterministic);

		struct BackpropResult {
			torch::Tensor actionLogProbs;
			torch::Tensor entropy; // mean over batch
		};
		BackpropResult GetBackpropData(torch::Tensor obs, torch::Tensor acts);
	};
}
