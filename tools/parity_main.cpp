// parity_main.cpp — Phase 2 parity gate executable.
//
// Reads the seeded obs + Python action-mean artifacts produced by tools/parity_gen.py,
// loads the SAME BC checkpoint (converted to .lt) into a C++ GaussianPolicy, runs the
// deterministic policy on the identical flat obs, and asserts max|delta| < 1e-4 against
// the Python means. Exits non-zero on mismatch so CI can gate on it.
//
// The actor (continuous) and FSP paths are INDEPENDENTLY gated by meta.txt flags. When the FSP
// flag is set, it builds a C++ FSPNet (its own EARLPerceiver trunk), loads PPO_FSP.lt, runs the
// FSP forward on the same obs, and asserts the same max|delta| < 1e-4 bar against py_fsp.bin.
#include <RLGymPPO_CPP/PPO/PPOLearnerConfig.h>
#include <private/RLGymPPO_CPP/PPO/GaussianPolicy.h> // GaussianPolicy + PerceiverNet + EARLPerceiver
#include <private/RLGymPPO_CPP/Model/FSPNet.h>        // FSPNet (auxiliary head)
#include <private/RLGymPPO_CPP/Util/StateDictIO.h>     // version-independent weight loader
#include <torch/torch.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <filesystem>
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

	// meta.txt: "<N> <obsSize> <queryFeatures> <kvFeatures> <nTokens> <nControls> <entityFeatures> <actorEnabled> <fspEnabled>"
	int64_t N, obsSize, queryFeatures, kvFeatures, nTokens, nControls, entityFeatures, actorEnabled, fspEnabled;
	{
		std::ifstream meta(dataFolder + "/meta.txt");
		meta >> N >> obsSize >> queryFeatures >> kvFeatures >> nTokens >> nControls
		     >> entityFeatures >> actorEnabled >> fspEnabled;
		if (!meta) { std::fprintf(stderr, "bad meta.txt\n"); return 2; }
	}

	auto obs = readFloats(dataFolder + "/obs.bin", (size_t)(N * obsSize));
	auto obsTensor = torch::from_blob(obs.data(), { N, obsSize },
		torch::TensorOptions().dtype(torch::kFloat32)).clone();

	// Continuous actor path (optional).
	if (actorEnabled) {
		auto pyMeans = readFloats(dataFolder + "/py_means.bin", (size_t)(N * nControls));

		PerceiverConfig cfg;
		cfg.nDims = 256; cfg.nHeads = 8; cfg.nLayers = 4;
		cfg.nControls = (int)nControls;
		cfg.queryFeatures = (int)queryFeatures; cfg.kvFeatures = (int)kvFeatures;
		cfg.nTokens = (int)nTokens; cfg.chunkSize = 100000; // batch all N in one forward

		GaussianPolicy policy(cfg, torch::kCPU);
		try {
			ApplyStateDict(*policy.net.ptr(), LoadStateDict(weightsFolder + "/PPO_POLICY.lt"));
		} catch (const std::exception& e) {
			std::fprintf(stderr, "state-dict load failed: %s\n", e.what());
			return 3;
		}

		torch::Tensor cppMeans;
		try {
			auto res = policy.GetAction(obsTensor, /*deterministic=*/true);
			cppMeans = res.action.to(torch::kFloat32); // (N, nControls)
		} catch (const std::exception& e) {
			std::fprintf(stderr, "forward failed: %s\n", e.what());
			return 4;
		}

		auto pyT = torch::from_blob(pyMeans.data(), { N, nControls },
			torch::TensorOptions().dtype(torch::kFloat32)).clone();
		auto delta = (cppMeans - pyT).abs();
		float maxAbs = delta.max().item<float>();
		float meanAbs = delta.mean().item<float>();
		std::printf("actor parity rows=%lld obsSize=%lld nControls=%lld max|delta|=%.3e mean|delta|=%.3e\n",
			(long long)N, (long long)obsSize, (long long)nControls, maxAbs, meanAbs);
		if (!(maxAbs < 1e-4f)) {
			std::fprintf(stderr, "ACTOR PARITY FAIL: max|delta|=%.3e >= 1e-4\n", maxAbs);
			return 1;
		}
	}

	// FSP auxiliary head path (optional). The FSP net builds its OWN earl trunk (the runtime
	// "shared trunk" optimization is irrelevant to per-tensor forward parity).
	if (fspEnabled) {
		FSPNet fsp(
			EARLPerceiver(256, 8, 4, queryFeatures, kvFeatures),
			entityFeatures, queryFeatures, kvFeatures, nTokens, 100000);
		try {
			ApplyStateDict(*fsp.ptr(), LoadStateDict(weightsFolder + "/PPO_FSP.lt"));
		} catch (const std::exception& e) {
			std::fprintf(stderr, "fsp state-dict load failed: %s\n", e.what());
			return 5;
		}

		auto pyFsp = readFloats(dataFolder + "/py_fsp.bin", (size_t)(N * entityFeatures));
		torch::Tensor cppFsp;
		try {
			cppFsp = fsp->Forward(obsTensor).to(torch::kFloat32); // (N, entityFeatures)
		} catch (const std::exception& e) {
			std::fprintf(stderr, "fsp forward failed: %s\n", e.what());
			return 6;
		}

		auto pyFspT = torch::from_blob(pyFsp.data(), { N, entityFeatures },
			torch::TensorOptions().dtype(torch::kFloat32)).clone();
		auto fspDelta = (cppFsp - pyFspT).abs();
		float fspMax = fspDelta.max().item<float>();
		float fspMean = fspDelta.mean().item<float>();
		std::printf("fsp parity rows=%lld entityFeatures=%lld max|delta|=%.3e mean|delta|=%.3e\n",
			(long long)N, (long long)entityFeatures, fspMax, fspMean);
		if (!(fspMax < 1e-4f)) {
			std::fprintf(stderr, "FSP PARITY FAIL: max|delta|=%.3e >= 1e-4\n", fspMax);
			return 1;
		}
	}

	std::printf("PARITY PASS\n");
	return 0;
}
