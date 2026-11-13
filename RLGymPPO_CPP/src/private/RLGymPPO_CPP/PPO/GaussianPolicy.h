#pragma once
#include "../Model/PerceiverNet.h"
#include <RLGymPPO_CPP/PPO/PPOLearnerConfig.h>

// Switchable actor wrapper. The PerceiverNet trunk is identical for both arms; only the
// head and the log_prob/entropy/sample math differ, mirroring reference.agent's
// get_actor() dispatch on spec.head:
//   CONTINUOUS (ContinuousPolicy): head -> [mean(nControls) || log_std(nControls)];
//     policy = Independent Normal. log_std.clamp(-5,2); std = exp(log_std).clamp(1e-4,10);
//     log_prob/entropy SUM over channels.
//   DISCRETE   (CategoricalControlsPolicy): head -> (nControls, maxBins) logits; one
//     independent softmax per channel (pad slots beyond nBins[c] masked to -inf); log_prob/
//     entropy are the Independent(Categorical) sum over channels; sample/argmax -> bin idx.
// The class name is kept as GaussianPolicy for call-site stability; it is the actor wrapper
// for BOTH arms (the head type is selected by config.headType).
namespace RLGPC {
	class GaussianPolicy {
	public:
		PerceiverNet net{nullptr};
		PerceiverConfig config;
		torch::Device device;
		int nControls;
		HeadType headType;

		GaussianPolicy(const PerceiverConfig& config, torch::Device device);

		RG_NO_COPY(GaussianPolicy);

		// Deep copy of this policy into a fresh GaussianPolicy (same layout config)
		GaussianPolicy* Clone();

		void CopyTo(GaussianPolicy& to) { net->CopyTo(to.net.ptr()); }

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
