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

RLGPC::CategoricalControlsHeadImpl::CategoricalControlsHeadImpl(int inFeatures, const std::vector<int>& nBins)
	: nBins(nBins) {
	maxBins = 0;
	for (int b : nBins) maxBins = std::max(maxBins, b);
	// Python: self.logits = Linear(in_features, n_controls * max_bins)
	logits = register_module("logits",
		torch::nn::Linear(torch::nn::LinearOptions(inFeatures, (int64_t)nBins.size() * maxBins)));
}

torch::Tensor RLGPC::CategoricalControlsHeadImpl::forward(const torch::Tensor& x) {
	// Python: out = self.logits(x); out.reshape(*out.shape[:-1], n_controls, max_bins)
	auto out = logits->forward(x); // (B,1,nControls*maxBins)
	return out.reshape({ out.size(0), out.size(1), (int64_t)nBins.size(), (int64_t)maxBins });
}

RLGPC::PerceiverNetImpl::PerceiverNetImpl(
	EARLPerceiver earl, torch::nn::Linear output,
	int64_t queryFeatures, int64_t kvFeatures, int64_t nTokens, int64_t chunkSize) :
	earl(std::move(earl)), criticHead(std::move(output)), headType(HeadType::CONTINUOUS), chunkSize(chunkSize),
	queryFeatures(queryFeatures), kvFeatures(kvFeatures), nTokens(nTokens) {

	register_module("earl", this->earl);
	register_module("output", this->criticHead);
}

RLGPC::PerceiverNetImpl::PerceiverNetImpl(
	EARLPerceiver earl, ContinuousControlsHead output,
	int64_t queryFeatures, int64_t kvFeatures, int64_t nTokens, int64_t chunkSize) :
	earl(std::move(earl)), contHead(std::move(output)), headType(HeadType::CONTINUOUS), chunkSize(chunkSize),
	queryFeatures(queryFeatures), kvFeatures(kvFeatures), nTokens(nTokens) {

	register_module("earl", this->earl);
	register_module("output", this->contHead);
}

RLGPC::PerceiverNetImpl::PerceiverNetImpl(
	EARLPerceiver earl, CategoricalControlsHead output,
	int64_t queryFeatures, int64_t kvFeatures, int64_t nTokens, int64_t chunkSize) :
	earl(std::move(earl)), catHead(std::move(output)), headType(HeadType::DISCRETE), chunkSize(chunkSize),
	queryFeatures(queryFeatures), kvFeatures(kvFeatures), nTokens(nTokens) {

	register_module("earl", this->earl);
	register_module("output", this->catHead);
}

torch::Tensor RLGPC::PerceiverNetImpl::Forward(const torch::Tensor& flatObs) {
	// flatObs: (B, obsSize) where obsSize = queryFeatures + nTokens*kvFeatures + nTokens + 1 (mode).
	// Rebuild the (q, kv, m, mode) tuple the Python agent received per-sample as
	// (1,1,Q)/(1,T,W)/(1,T)/(1,): q (B,1,Q) kv (B,T,W) m (B,T) mode (B,) int64.
	const int64_t B = flatObs.size(0);
	const int64_t kvEnd = queryFeatures + nTokens * kvFeatures;
	const int64_t maskEnd = kvEnd + nTokens; // mode is the single column at [maskEnd]

	auto q = flatObs.slice(1, 0, queryFeatures).view({ B, 1, queryFeatures });
	auto kv = flatObs.slice(1, queryFeatures, kvEnd).view({ B, nTokens, kvFeatures });
	auto m = flatObs.slice(1, kvEnd, maskEnd);
	auto mode = flatObs.slice(1, maskEnd, maskEnd + 1).squeeze(1).to(torch::kLong);

	// Chunked forward (Python PerceiverNet.forward chunks along batch to bound VRAM; the
	// critic runs over the whole rollout buffer in one call).
	if (B <= chunkSize) {
		auto res = earl->forward(q, kv, m, mode);
		res = headType == HeadType::DISCRETE ? catHead->forward(torch::relu(res))
			: (criticHead ? criticHead->forward(torch::relu(res)) : contHead->forward(torch::relu(res)));
		return res.slice(1, 0, 1).squeeze(1); // [:, 0, :]
	}

	std::vector<torch::Tensor> outs;
	for (int64_t i = 0; i < B; i += chunkSize) {
		int64_t end = std::min(i + chunkSize, B);
		auto res = earl->forward(q.slice(0, i, end), kv.slice(0, i, end), m.slice(0, i, end),
			mode.slice(0, i, end));
		res = headType == HeadType::DISCRETE ? catHead->forward(torch::relu(res))
			: (criticHead ? criticHead->forward(torch::relu(res)) : contHead->forward(torch::relu(res)));
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
