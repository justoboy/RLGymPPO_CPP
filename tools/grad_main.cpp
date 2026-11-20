// grad_main.cpp — P3 league gradient-direction parity gate (C++ side).
//
// Reads a seeded fixture (flat obs + actions + old log-probs + advantages) produced by
// tools/grad_fixture.py, loads the SAME BC actor weights (PPO_POLICY.lt) into a C++ GaussianPolicy,
// runs the EXACT PPO policy-loss backward that PPOLearner::Learn performs, then writes the resulting
// per-parameter gradients (keyed by qualified name, sorted) so the Python side can diff them.
//
// This gates the per-member learn path: the league routes each episode's experience to its owner's
// buffer and runs this same update on it. If the C++ policy gradient matches Python's elementwise
// (cosine ~1.0) on a frozen batch, the routing's per-member update math is correct.
#include <RLGymPPO_CPP/PPO/PPOLearnerConfig.h>
#include <private/RLGymPPO_CPP/PPO/GaussianPolicy.h>
#include <private/RLGymPPO_CPP/Util/StateDictIO.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace RLGPC;

static std::vector<float> readFloats(const std::string& path, size_t count) {
	std::ifstream in(path, std::ios::binary);
	if (!in) { std::fprintf(stderr, "missing %s\n", path.c_str()); std::exit(2); }
	std::vector<float> v(count);
	in.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(count * sizeof(float)));
	if (!in) { std::fprintf(stderr, "short read %s\n", path.c_str()); std::exit(2); }
	return v;
}

int main(int argc, char** argv) {
	if (argc < 3) {
		std::fprintf(stderr, "usage: %s <data_folder> <weights_folder>\n", argv[0]);
		return 2;
	}
	const std::string dataFolder = argv[1];
	const std::string weightsFolder = argv[2];

	// meta.txt: "<N> <obsSize> <queryFeatures> <kvFeatures> <nTokens> <nControls> <entityFeatures> <clipRange> <entCoef>"
	int64_t N, obsSize, queryFeatures, kvFeatures, nTokens, nControls, entityFeatures;
	float clipRange, entCoef;
	{
		std::ifstream meta(dataFolder + "/meta.txt");
		meta >> N >> obsSize >> queryFeatures >> kvFeatures >> nTokens >> nControls
		     >> entityFeatures >> clipRange >> entCoef;
		if (!meta) { std::fprintf(stderr, "bad meta.txt\n"); return 2; }
	}

	auto obs = readFloats(dataFolder + "/obs.bin", (size_t)(N * obsSize));
	auto acts = readFloats(dataFolder + "/acts.bin", (size_t)(N * nControls));
	auto oldlp = readFloats(dataFolder + "/oldlp.bin", (size_t)N);
	auto adv = readFloats(dataFolder + "/adv.bin", (size_t)N);

	auto obsT = torch::from_blob(obs.data(), { N, obsSize }, torch::TensorOptions().dtype(torch::kFloat32)).clone();
	auto actsT = torch::from_blob(acts.data(), { N, nControls }, torch::TensorOptions().dtype(torch::kFloat32)).clone();
	auto oldlpT = torch::from_blob(oldlp.data(), { N }, torch::TensorOptions().dtype(torch::kFloat32)).clone();
	auto advT = torch::from_blob(adv.data(), { N }, torch::TensorOptions().dtype(torch::kFloat32)).clone();

	PerceiverConfig cfg;
	cfg.nDims = 256; cfg.nHeads = 8; cfg.nLayers = 4;
	cfg.nControls = (int)nControls;
	cfg.queryFeatures = (int)queryFeatures; cfg.kvFeatures = (int)kvFeatures;
	cfg.nTokens = (int)nTokens; cfg.chunkSize = 100000;

	GaussianPolicy policy(cfg, torch::kCPU);
	try {
		ApplyStateDict(*policy.net.ptr(), LoadStateDict(weightsFolder + "/PPO_POLICY.lt"));
	} catch (const std::exception& e) {
		std::fprintf(stderr, "state-dict load failed: %s\n", e.what());
		return 3;
	}

	// Mirror PPOLearner::Learn's policy-loss math exactly (single batch, batchSizeRatio == 1).
	torch::Tensor logProbs, entropy;
	{
		auto bp = policy.GetBackpropData(obsT, actsT);
		logProbs = bp.actionLogProbs; // (N,)
		entropy = bp.entropy;         // scalar (already mean over batch)
	}
	auto ratio = torch::exp(logProbs - oldlpT);
	auto clipped = torch::clamp(ratio, 1 - clipRange, 1 + clipRange);
	auto policyLoss = -torch::min(ratio * advT, clipped * advT).mean();
	auto ppoLoss = policyLoss - entropy * entCoef;
	ppoLoss.backward();

	// Collect gradients keyed by qualified parameter name, sorted, so ordering is unambiguous.
	std::map<std::string, torch::Tensor> grads;
	for (auto p : policy.net->named_parameters()) {
		grads[p.key()] = p.value().grad().detach().to(torch::kFloat32).flatten().cpu();
	}

	// Write keys.txt (one qualified name per line, sorted) + cpp_grads.bin (concatenated in that order).
	std::vector<std::string> keys;
	keys.reserve(grads.size());
	for (auto& kv : grads) keys.push_back(kv.first);
	{
		std::ofstream kf(dataFolder + "/cpp_keys.txt");
		for (auto& k : keys) kf << k << "\n";
	}
	std::ofstream gf(dataFolder + "/cpp_grads.bin", std::ios::binary);
	for (auto& k : keys) {
		auto flat = grads[k].contiguous();
		gf.write(reinterpret_cast<const char*>(flat.data_ptr<float>()),
		         (std::streamsize)(flat.numel() * sizeof(float)));
	}
	std::printf("grad cpp params=%zu N=%lld obsSize=%lld\n", grads.size(), (long long)N, (long long)obsSize);
	std::printf("GRAD CPP OK\n");
	return 0;
}
