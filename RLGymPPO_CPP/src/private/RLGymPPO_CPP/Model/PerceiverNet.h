#pragma once
#include "EARLPerceiver.h"
#include "../../../public/RLGymPPO_CPP/PPO/PPOLearnerConfig.h" // HeadType, PerceiverConfig

// Reference-implementation parity. Module/parameter names match the reference state dicts
// 1:1 (after the converter strips the actor's "net." prefix):
//   actor:  earl.*  output.mean.weight output.mean.bias output.log_std
//   critic: earl.*  output.weight output.bias
// Forward() takes the FLAT per-player obs (q || kv || m, obsSize wide) and rebuilds the
// (q, kv, m) tuple internally, so the buffer/trajectory plumbing stays flat.

namespace RLGPC {
	// Continuous head: (B,1,256) -> (B,1,2*nControls) = [mean, log_std].
	class ContinuousControlsHeadImpl : public torch::nn::Module {
	public:
		torch::nn::Linear mean{nullptr};
		torch::Tensor log_std; // (nControls,) learnable, init -0.5

		ContinuousControlsHeadImpl(int inFeatures, int nControls, float initLogStd = -0.5f);

		torch::Tensor forward(const torch::Tensor& x);
	};

	TORCH_MODULE(ContinuousControlsHead);

	// Python CategoricalControlsHead: (B,1,256) -> (B,1,nControls,max_bins). One independent
	// softmax per control channel (the discrete arm of the BC representation A/B). The logits
	// are padded to the widest channel (max_bins) and sliced per channel at decode time.
	class CategoricalControlsHeadImpl : public torch::nn::Module {
	public:
		torch::nn::Linear logits{nullptr};
		std::vector<int> nBins; // per-channel class counts (padded to max_bins)
		int maxBins;

		CategoricalControlsHeadImpl(int inFeatures, const std::vector<int>& nBins);

		torch::Tensor forward(const torch::Tensor& x);
	};

	TORCH_MODULE(CategoricalControlsHead);

	class PerceiverNetImpl : public torch::nn::Module {
	public:
		EARLPerceiver earl{nullptr};

		// Exactly ONE head is registered (as "output"); the others stay unregistered so their
		// params never enter the state dict and the loaded dict keys match exactly.
		torch::nn::Linear criticHead{nullptr};          // critic: output.weight / output.bias
		ContinuousControlsHead contHead{nullptr};       // actor (cont):  output.mean.* / output.log_std
		CategoricalControlsHead catHead{nullptr};       // actor (disc): output.logits.weight / .bias
		HeadType headType;

		// Obs layout: flat obs = q(queryFeatures) ++ kv(nTokens*kvFeatures) ++ m(nTokens)
		int64_t chunkSize;
		int64_t queryFeatures;
		int64_t kvFeatures;
		int64_t nTokens;

		// Critic ctor (output = Linear(256,1))
		PerceiverNetImpl(EARLPerceiver earl, torch::nn::Linear output,
			int64_t queryFeatures, int64_t kvFeatures, int64_t nTokens, int64_t chunkSize = 2048);
		// Continuous actor ctor (output = ContinuousControlsHead)
		PerceiverNetImpl(EARLPerceiver earl, ContinuousControlsHead output,
			int64_t queryFeatures, int64_t kvFeatures, int64_t nTokens, int64_t chunkSize = 2048);
		// Discrete actor ctor (output = CategoricalControlsHead)
		PerceiverNetImpl(EARLPerceiver earl, CategoricalControlsHead output,
			int64_t queryFeatures, int64_t kvFeatures, int64_t nTokens, int64_t chunkSize = 2048);

		// One (B, obsSize) flat batch -> head output. CONTINUOUS actor -> (B, 2*nControls);
		// DISCRETE actor -> (B, nControls, maxBins); critic -> (B, 1).
		torch::Tensor Forward(const torch::Tensor& flatObs);

		void CopyTo(std::shared_ptr<PerceiverNetImpl> to);
	};

	TORCH_MODULE(PerceiverNet);
}
