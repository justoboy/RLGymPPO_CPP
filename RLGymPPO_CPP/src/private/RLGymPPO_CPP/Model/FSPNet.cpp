#include "FSPNet.h"
#include <private/RLGymPPO_CPP/FrameworkTorch.h>
#include <algorithm>

// Mirrors reference.fsp.FSPNet._forward:
//   res = self.earl(q, kv, m)            # (B, 1, 256)
//   return self.head(self.relu(res))[:, 0, :]   # (B, entityFeatures)
// Chunked along the batch (Python FSPNet.forward chunks to bound peak VRAM on full-batch calls).

RLGPC::FSPNetImpl::FSPNetImpl(
	EARLPerceiver earl, int64_t entityFeatures,
	int64_t queryFeatures, int64_t kvFeatures, int64_t nTokens, int64_t chunkSize) :
	earl(std::move(earl)), chunkSize(chunkSize),
	queryFeatures(queryFeatures), kvFeatures(kvFeatures), nTokens(nTokens) {

	register_module("earl", this->earl);
	// Python: self.head = nn.Linear(256, out_features); registered as "head".
	head = register_module("head",
		torch::nn::Linear(torch::nn::LinearOptions(this->earl->nDims, entityFeatures)));
}

torch::Tensor RLGPC::FSPNetImpl::Forward(const torch::Tensor& flatObs) {
	// flatObs: (B, obsSize) where obsSize = queryFeatures + nTokens*kvFeatures + nTokens.
	const int64_t B = flatObs.size(0);
	const int64_t kvEnd = queryFeatures + nTokens * kvFeatures;

	auto q = flatObs.slice(1, 0, queryFeatures).view({ B, 1, queryFeatures });
	auto kv = flatObs.slice(1, queryFeatures, kvEnd).view({ B, nTokens, kvFeatures });
	auto m = flatObs.slice(1, kvEnd, kvEnd + nTokens);

	if (B <= chunkSize) {
		auto res = earl->forward(q, kv, m);
		res = head->forward(torch::relu(res)); // (B, 1, entityFeatures)
		return res.slice(1, 0, 1).squeeze(1);  // [:, 0, :] -> (B, entityFeatures)
	}

	std::vector<torch::Tensor> outs;
	for (int64_t i = 0; i < B; i += chunkSize) {
		int64_t end = std::min(i + chunkSize, B);
		auto res = earl->forward(q.slice(0, i, end), kv.slice(0, i, end), m.slice(0, i, end));
		res = head->forward(torch::relu(res));
		outs.push_back(res.slice(1, 0, 1).squeeze(1));
	}
	return torch::cat(outs, 0);
}
