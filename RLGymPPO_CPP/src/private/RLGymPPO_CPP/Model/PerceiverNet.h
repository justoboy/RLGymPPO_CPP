#pragma once
#include "EARLPerceiver.h"

// Mirrors reference.agent.ContinuousControlsHead and reference.agent.PerceiverNet.
// Module/parameter names match the Python state dicts 1:1 (after the converter strips
// the actor's "net." prefix):
//   actor:  earl.*  output.mean.weight output.mean.bias output.log_std
//   critic: earl.*  output.weight output.bias
// Forward() takes the FLAT per-player obs (q || kv || m, obsSize wide) and rebuilds the
// (q, kv, m) tuple internally, so the buffer/trajectory plumbing stays flat.

namespace RLGPC {
	// Python ContinuousControlsHead: (B,1,256) -> (B,1,2*nControls) = [mean, log_std].
	class ContinuousControlsHeadImpl : public torch::nn::Module {
	public:
		torch::nn::Linear mean{nullptr};
		torch::Tensor log_std; // (nControls,) learnable, init -0.5

		ContinuousControlsHeadImpl(int inFeatures, int nControls, float initLogStd = -0.5f);

		torch::Tensor forward(const torch::Tensor& x);
	};

	TORCH_MODULE(ContinuousControlsHead);

	class PerceiverNetImpl : public torch::nn::Module {
	public:
		EARLPerceiver earl{nullptr};

		// Exactly ONE head is registered (as "output"); the other stays unregistered so its
		// params never enter the state dict and the loaded dict keys match exactly.
		torch::nn::Linear criticHead{nullptr};      // critic: output.weight / output.bias
		ContinuousControlsHead actorHead{nullptr};  // actor:  output.mean.* / output.log_std
		bool isActor;

		// Obs layout: flat obs = q(queryFeatures) ++ kv(nTokens*kvFeatures) ++ m(nTokens)
		int64_t chunkSize;
		int64_t queryFeatures;
		int64_t kvFeatures;
		int64_t nTokens;

		// Critic ctor (output = Linear(256,1))
		PerceiverNetImpl(EARLPerceiver earl, torch::nn::Linear output,
			int64_t queryFeatures, int64_t kvFeatures, int64_t nTokens, int64_t chunkSize = 2048);
		// Actor ctor (output = ContinuousControlsHead)
		PerceiverNetImpl(EARLPerceiver earl, ContinuousControlsHead output,
			int64_t queryFeatures, int64_t kvFeatures, int64_t nTokens, int64_t chunkSize = 2048);

		// One (B, obsSize) flat batch -> (B, outDim) head output (18 for actor, 1 for critic)
		torch::Tensor Forward(const torch::Tensor& flatObs);

		void CopyTo(std::shared_ptr<PerceiverNetImpl> to);
	};

	TORCH_MODULE(PerceiverNet);
}
