#include "GaussianPolicy.h"

#include <private/RLGymPPO_CPP/FrameworkTorch.h>

// Exact port of reference.agent.ContinuousPolicy:
//   mean, log_std = head_out.split(nControls); log_std.clamp(-5, 2)
//   std = exp(log_std).clamp(1e-4, 10)
//   sample: action = mean + std * eps
//   log_prob = Normal(action; mean, std).log_prob summed over channels (Independent, dim=-1)
//   entropy  = Normal.entropy summed over channels; Normal.entropy = log_std + 0.5*(1+log(2*pi))

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

RLGPC::GaussianPolicy::GaussianPolicy(const PerceiverConfig& config, torch::Device device)
	: config(config), device(device), nControls(config.nControls) {

	EARLPerceiver earl = EARLPerceiver(
		config.nDims, config.nHeads, config.nLayers, config.queryFeatures, config.kvFeatures);
	ContinuousControlsHead head = ContinuousControlsHead(config.nDims, config.nControls);
	net = PerceiverNet(earl, head, config.queryFeatures, config.kvFeatures, config.nTokens, config.chunkSize);
	net->to(device, true);
}

RLGPC::GaussianPolicy* RLGPC::GaussianPolicy::Clone() {
	GaussianPolicy* clone = new GaussianPolicy(config, device);
	net->CopyTo(clone->net);
	return clone;
}

RLGPC::GaussianPolicy::ActionResult RLGPC::GaussianPolicy::GetAction(torch::Tensor obs, bool deterministic) {
	RG_NOGRAD;
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
	acts = acts.to(torch::kFloat32, true);
	auto out = net->Forward(obs); // (B, 2*nControls)
	auto mean = out.slice(1, 0, nControls);
	auto logStd = out.slice(1, nControls, 2 * nControls).clamp(-5.0f, 2.0f);
	auto std = torch::exp(logStd).clamp(1e-4f, 10.0f);

	auto actionLogProbs = _channel_log_prob(acts, mean, std);
	auto entropy = _channel_entropy(std);

	return BackpropResult{ actionLogProbs.to(device, true), entropy.to(device).mean() };
}
