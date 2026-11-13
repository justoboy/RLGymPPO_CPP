#include "PerceiverNet.h"
#include <private/RLGymPPO_CPP/FrameworkTorch.h>
#include <algorithm>

RLGPC::ContinuousControlsHeadImpl::ContinuousControlsHeadImpl(int inFeatures, int nControls, float initLogStd) {
	mean = register_module("mean", torch::nn::Linear(torch::nn::LinearOptions(inFeatures, nControls)));
	// Python: self.log_std = nn.Parameter(torch.full((n_controls,), init_log_std))
	log_std = register_parameter("log_std", torch::full({ (int64_t)nControls }, initLogStd));
}

torch::Tensor RLGPC::ContinuousControlsHeadImpl::forward(const torch::Tensor& x) {
	// Python: mean = self.mean(x); log_std.expand(*mean.shape[:-1], -1); cat([mean, log_std], -1)
	auto m = mean->forward(x);
	auto ls = log_std.expand({ m.size(0), m.size(1), m.size(2) });
	return torch::cat({ m, ls }, -1);
}

RLGPC::PerceiverNetImpl::PerceiverNetImpl(
	EARLPerceiver earl, torch::nn::Linear output,
	int64_t queryFeatures, int64_t kvFeatures, int64_t nTokens, int64_t chunkSize) :
	earl(std::move(earl)), criticHead(std::move(output)), isActor(false), chunkSize(chunkSize),
	queryFeatures(queryFeatures), kvFeatures(kvFeatures), nTokens(nTokens) {

	register_module("earl", this->earl);
	register_module("output", this->criticHead);
}

RLGPC::PerceiverNetImpl::PerceiverNetImpl(
	EARLPerceiver earl, ContinuousControlsHead output,
	int64_t queryFeatures, int64_t kvFeatures, int64_t nTokens, int64_t chunkSize) :
	earl(std::move(earl)), actorHead(std::move(output)), isActor(true), chunkSize(chunkSize),
	queryFeatures(queryFeatures), kvFeatures(kvFeatures), nTokens(nTokens) {

	register_module("earl", this->earl);
	register_module("output", this->actorHead);
}

torch::Tensor RLGPC::PerceiverNetImpl::Forward(const torch::Tensor& flatObs) {
	// flatObs: (B, obsSize) where obsSize = queryFeatures + nTokens*kvFeatures + nTokens.
	// Rebuild the (q, kv, m) tuple the Python agent received per-sample as
	// (1,1,Q)/(1,T,W)/(1,T): q (B,1,Q) kv (B,T,W) m (B,T).
	const int64_t B = flatObs.size(0);
	const int64_t kvEnd = queryFeatures + nTokens * kvFeatures;

	auto q = flatObs.slice(1, 0, queryFeatures).view({ B, 1, queryFeatures });
	auto kv = flatObs.slice(1, queryFeatures, kvEnd).view({ B, nTokens, kvFeatures });
	auto m = flatObs.slice(1, kvEnd, kvEnd + nTokens);

	// Chunked forward (Python PerceiverNet.forward chunks along batch to bound VRAM; the
	// critic runs over the whole rollout buffer in one call).
	if (B <= chunkSize) {
		auto res = earl->forward(q, kv, m);
		res = isActor ? actorHead->forward(torch::relu(res)) : criticHead->forward(torch::relu(res));
		return res.slice(1, 0, 1).squeeze(1); // [:, 0, :]
	}

	std::vector<torch::Tensor> outs;
	for (int64_t i = 0; i < B; i += chunkSize) {
		int64_t end = std::min(i + chunkSize, B);
		auto res = earl->forward(q.slice(0, i, end), kv.slice(0, i, end), m.slice(0, i, end));
		res = isActor ? actorHead->forward(torch::relu(res)) : criticHead->forward(torch::relu(res));
		outs.push_back(res.slice(1, 0, 1).squeeze(1));
	}
	return torch::cat(outs, 0);
}

void RLGPC::PerceiverNetImpl::CopyTo(std::shared_ptr<PerceiverNetImpl> to) {
	RG_NOGRAD;
	// Parameter registration order is identical (earl.* then output.*) between both
	// instances, so a positional copy is exact.
	auto fromParams = this->parameters();
	auto toParams = to->parameters();
	assert(fromParams.size() == toParams.size());
	for (size_t i = 0; i < fromParams.size(); i++)
		toParams[i].copy_(fromParams[i], true);
}
