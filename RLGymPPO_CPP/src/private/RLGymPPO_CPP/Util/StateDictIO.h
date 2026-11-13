#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

// Version-independent state-dict serialization.
//
// torch's own pickle archive format is NOT stable across versions (a .pt written by
// torch 2.14 embeds constants.pkl that torch 2.2.0's reader rejects). To make the BC
// weights portable across the python (2.14) -> C++ libtorch (2.2.0) boundary, the
// converter writes a tiny self-describing binary blob instead, and we copy each tensor
// into the module by qualified name. Format (little-endian):
//   uint32 magic  = 0x57454947 ("WEIG")
//   uint32 version = 1
//   uint32 num_tensors
//   per tensor:
//     uint32 key_len ; char[key_len] key
//     uint32 rank    ; uint32 dims[rank]
//     float32 data[prod(dims)]
namespace RLGPC {
	inline constexpr uint32_t RG_SD_MAGIC = 0x57454947u;
	inline constexpr uint32_t RG_SD_VERSION = 1u;

	inline std::unordered_map<std::string, torch::Tensor> LoadStateDict(const std::string& path) {
		std::ifstream in(path, std::ios::binary);
		if (!in) throw std::runtime_error("LoadStateDict: cannot open " + path);

		auto rdU32 = [&](void) -> uint32_t {
			uint32_t v; in.read(reinterpret_cast<char*>(&v), 4);
			if (!in) throw std::runtime_error("LoadStateDict: truncated header in " + path);
			return v;
			};
		auto rdF32 = [&](float* dst, size_t n) {
			in.read(reinterpret_cast<char*>(dst), (std::streamsize)(n * 4));
			if (!in) throw std::runtime_error("LoadStateDict: truncated data in " + path);
			};

		if (rdU32() != RG_SD_MAGIC) throw std::runtime_error("LoadStateDict: bad magic in " + path);
		if (rdU32() != RG_SD_VERSION) throw std::runtime_error("LoadStateDict: bad version in " + path);
		const uint32_t num = rdU32();

		std::unordered_map<std::string, torch::Tensor> out;
		for (uint32_t t = 0; t < num; t++) {
			const uint32_t keyLen = rdU32();
			std::string key(keyLen, '\0');
			in.read(key.data(), keyLen);
			if (!in) throw std::runtime_error("LoadStateDict: truncated key in " + path);

			const uint32_t rank = rdU32();
			std::vector<int64_t> dims(rank);
			uint32_t numel = 1;
			for (uint32_t d = 0; d < rank; d++) { dims[d] = (int64_t)rdU32(); numel *= (uint32_t)dims[d]; }

			std::vector<float> data(numel);
			rdF32(data.data(), numel);

			auto opt = torch::TensorOptions().dtype(torch::kFloat32);
			out.emplace(std::move(key), torch::from_blob(data.data(), dims, opt).clone());
		}
		return out;
	}

	// Copies every tensor from the map into the module's matching parameter/buffer by
	// qualified name. Throws if a key is missing or a shape mismatches.
	inline void ApplyStateDict(torch::nn::Module& module,
		const std::unordered_map<std::string, torch::Tensor>& sd) {
		torch::NoGradGuard noGrad;
		for (const auto& kv : module.named_parameters()) {
			auto it = sd.find(kv.key());
			if (it == sd.end()) throw std::runtime_error("ApplyStateDict: missing param " + kv.key());
			if (it->second.sizes() != kv.value().sizes())
				throw std::runtime_error("ApplyStateDict: shape mismatch at " + kv.key());
			kv.value().copy_(it->second.to(kv.value().device()));
		}
		for (const auto& kv : module.named_buffers()) {
			auto it = sd.find(kv.key());
			if (it == sd.end()) continue; // buffers optional
			kv.value().copy_(it->second.to(kv.value().device()));
		}
	}
}
