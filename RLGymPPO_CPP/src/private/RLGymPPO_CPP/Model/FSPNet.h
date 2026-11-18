#pragma once
#include "EARLPerceiver.h"
#include "../../../public/RLGymPPO_CPP/PPO/PPOLearnerConfig.h" // PerceiverConfig

// FSP auxiliary head. It shares the SAME EARLPerceiver trunk instance the actor uses (the
// shared handle is passed in), so the auxiliary gradient flows into the encoder the actor uses.
// Module/parameter names match the reference fsp_state_dict 1:1:
//   fsp:  earl.*  head.weight head.bias
// Forward() takes the FLAT per-player obs (q || kv || m, obsSize wide) and rebuilds the
// (q, kv, m) tuple internally, chunked along the batch to bound VRAM (exactly like PerceiverNet).
// Output: (B, entityFeatures) = head(relu(earl(q,kv,m)))[:, 0, :].

namespace RLGPC {
	class FSPNetImpl : public torch::nn::Module {
	public:
		// Shared with the actor: the SAME EARLPerceiver handle instance (shared weights).
		EARLPerceiver earl{nullptr};
		torch::nn::Linear head{nullptr}; // Linear(256, entityFeatures)

		int64_t chunkSize;
		int64_t queryFeatures;
		int64_t kvFeatures;
		int64_t nTokens;

		FSPNetImpl(EARLPerceiver earl, int64_t entityFeatures,
			int64_t queryFeatures, int64_t kvFeatures, int64_t nTokens, int64_t chunkSize = 2048);

		// One (B, obsSize) flat batch -> (B, entityFeatures) next-frame self-entity prediction.
		torch::Tensor Forward(const torch::Tensor& flatObs);
	};

	TORCH_MODULE(FSPNet);
}
