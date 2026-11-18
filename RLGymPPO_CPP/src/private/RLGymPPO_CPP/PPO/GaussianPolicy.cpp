#include "GaussianPolicy.h"

#include <private/RLGymPPO_CPP/FrameworkTorch.h>

// CONTINUOUS arm (reference-implementation parity):
//   mean, log_std = head_out.split(nControls); log_std.clamp(-5, 2)
//   std = exp(log_std).clamp(1e-4, 10)
//   sample: action = mean + std * eps
//   log_prob = Normal(action; mean, std).log_prob summed over channels (Independent, dim=-1)
//   entropy  = Normal.entropy summed over channels; Normal.entropy = log_std + 0.5*(1+log(2*pi))
//
// DISCRETE arm (reference-implementation parity):
//   logits (B, nControls, maxBins); pad slots beyond nBins[c] masked to -inf; probs=softmax,
//   logProbs=log_softmax. sample -> per-channel categorical bin index; deterministic -> argmax.
//   log_prob = sum_c logProbs[c, idx_c]; entropy = sum_c -(p_c * logp_c).

static inline torch::Tensor _channel_log_prob(
	const torch::Tensor& action, const torch::Tensor& mean, const torch::Tensor& std) {
	// Normal.log_prob = -0.5*((x-mu)/sigma)^2 - log(sigma) - 0.5*log(2*pi)
	constexpr float HALF_LOG_2PI = 0.9189385332046727f; // 0.5*log(2*pi)
	auto lp = -0.5f * ((action - mean) / std).pow(2)
		- torch::log(std) - HALF_LOG_2PI;
	return lp.sum(1); // Independent(reinterpreted_batch_ndims=1) -> sum channels
}

static inline torch::Tensor _channel_entropy(const torch::Tensor& std) {
	constexpr float HALF_LOG_2PI_E = 1.4189385332046727f; // 0.5*(1+log(2*pi))
	return (torch::log(std) + HALF_LOG_2PI_E).sum(1);
}

// Build the per-channel pad mask (B, nControls, maxBins): 0 at valid bins, -inf at pad cols.
static inline torch::Tensor _pad_mask(const std::vector<int>& nBins, int64_t maxBins, torch::Device device) {
	const int64_t nC = (int64_t)nBins.size();
	auto mask = torch::zeros({ nC, maxBins }, torch::TensorOptions().dtype(torch::kFloat32).device(device));
	for (int64_t c = 0; c < nC; c++)
		if (nBins[c] < maxBins)
			mask.index_put_({ c, torch::indexing::Slice(nBins[c], maxBins) }, (float)-INFINITY);
	return mask;
}

RLGPC::GaussianPolicy::GaussianPolicy(const PerceiverConfig& config, torch::Device device)
	: config(config), device(device), nControls(config.nControls), headType(config.headType) {

	EARLPerceiver earl = EARLPerceiver(
		config.nDims, config.nHeads, config.nLayers, config.queryFeatures, config.kvFeatures);

	if (config.headType == HeadType::DISCRETE) {
		CategoricalControlsHead head = CategoricalControlsHead(config.nDims, config.nBins);
		net = PerceiverNet(earl, head, config.queryFeatures, config.kvFeatures, config.nTokens, config.chunkSize);
	} else {
		ContinuousControlsHead head = ContinuousControlsHead(config.nDims, config.nControls);
		net = PerceiverNet(earl, head, config.queryFeatures, config.kvFeatures, config.nTokens, config.chunkSize);
	}
	net->to(device, true);
}

RLGPC::GaussianPolicy* RLGPC::GaussianPolicy::Clone() {
	GaussianPolicy* clone = new GaussianPolicy(config, device);
	net->CopyTo(clone->net.ptr());
	return clone;
}

RLGPC::GaussianPolicy::ActionResult RLGPC::GaussianPolicy::GetAction(torch::Tensor obs, bool deterministic) {
	RG_NOGRAD;

	if (headType == HeadType::DISCRETE) {
		auto logits = net->Forward(obs); // (B, nControls, maxBins)
		auto mask = _pad_mask(config.nBins, logits.size(2), logits.device());
		auto logProbs = torch::log_softmax(logits + mask, -1); // (B, nControls, maxBins)

		torch::Tensor idx; // (B, nControls) bin indices
		if (deterministic) {
			idx = logits.argmax(-1);
		} else {
			int64_t B = logits.size(0), nC = logits.size(1), maxB = logits.size(2);
			auto probs = torch::softmax(logits + mask, -1).reshape({ B * nC, maxB });
			idx = torch::multinomial(probs, 1, true).reshape({ B, nC });
		}
		auto idxG = idx.unsqueeze(-1);
		auto chosenLogProb = logProbs.gather(2, idxG).squeeze(2).sum(1); // (B,)
		return ActionResult{ idx.to(torch::kFloat32).cpu(), chosenLogProb.cpu() };
	}

	auto out = net->Forward(obs); // (B, 2*nControls) = [mean || log_std]
	auto mean = out.slice(1, 0, nControls);
	auto logStd = out.slice(1, nControls, 2 * nControls).clamp(-5.0f, 2.0f);
	auto std = torch::exp(logStd).clamp(1e-4f, 10.0f);

	if (deterministic) {
		auto action = mean.cpu();
		return ActionResult{ action, torch::zeros({ action.size(0) }) };
	}

	auto eps = torch::randn_like(mean);
	auto action = mean + std * eps;
	auto logProb = _channel_log_prob(action, mean, std);
	return ActionResult{ action.cpu(), logProb.cpu() };
}

RLGPC::GaussianPolicy::BackpropResult RLGPC::GaussianPolicy::GetBackpropData(torch::Tensor obs, torch::Tensor acts) {
	// NOTE: No NoGradGuard here — this is the differentiable pass feeding the PPO loss.

	if (headType == HeadType::DISCRETE) {
		auto logits = net->Forward(obs); // (B, nControls, maxBins)
		auto mask = _pad_mask(config.nBins, logits.size(2), logits.device());
		auto logProbs = torch::log_softmax(logits + mask, -1);
		auto probs = torch::softmax(logits + mask, -1);

		auto idx = acts.to(torch::kLong, true).view({ logits.size(0), logits.size(1) });
		auto idxG = idx.unsqueeze(-1);
		auto actionLogProbs = logProbs.gather(2, idxG).squeeze(2).sum(1); // (B,)
		auto entropy = -(probs * logProbs).sum(-1).sum(1);                // (B,)

		return BackpropResult{ actionLogProbs.to(device, true), entropy.to(device).mean() };
	}

	acts = acts.to(torch::kFloat32, true);
	auto out = net->Forward(obs); // (B, 2*nControls)
	auto mean = out.slice(1, 0, nControls);
	auto logStd = out.slice(1, nControls, 2 * nControls).clamp(-5.0f, 2.0f);
	auto std = torch::exp(logStd).clamp(1e-4f, 10.0f);

	auto actionLogProbs = _channel_log_prob(acts, mean, std);
	auto entropy = _channel_entropy(std);

	return BackpropResult{ actionLogProbs.to(device, true), entropy.to(device).mean() };
}
