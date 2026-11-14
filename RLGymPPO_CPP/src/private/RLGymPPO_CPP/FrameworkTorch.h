#pragma once
#include <RLGymPPO_CPP/Framework.h>
#include <RLGymPPO_CPP/Lists.h>

// Include torch
#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <torch/utils.h>

#define RG_NOGRAD torch::NoGradGuard _noGradGuard

#define RG_AUTOCAST_ON() { \
at::autocast::set_enabled(true); \
at::autocast::set_autocast_gpu_dtype(torch::kBFloat16); \
at::autocast::set_autocast_cpu_dtype(torch::kFloat); \
}

#define RG_AUTOCAST_OFF() { \
at::autocast::clear_cache(); \
at::autocast::set_enabled(false); \
}

#define RG_HALFPERC_TYPE torch::ScalarType::BFloat16

namespace RLGPC {
	// Convert a (rows, cols) nested float list into a (rows, cols) float32 tensor.
	//
	// The previous implementation allocated a zero tensor and then issued one
	// slice()/copy_ ATen dispatch PER ROW, which is dispatch-bound (the C++ analogue of
	// the old reference obs-builder bottleneck). Since every row is contiguous and the
	// output is row-major contiguous, we copy all rows into one contiguous buffer and
	// build the tensor in a single from_blob().clone(). The float32 values are identical
	// (torch::equal == true); only the number of ATen dispatches changes.
	inline torch::Tensor FLIST2_TO_TENSOR(const FList2& list) {
		int rows = (int)list.size();
		int innerSize = (rows > 0) ? (int)list[0].size() : 0;

		std::vector<float> buf;
		buf.reserve((size_t)rows * (size_t)innerSize);
		for (auto& row : list)
			buf.insert(buf.end(), row.begin(), row.end());

		auto options = torch::TensorOptions().dtype(at::kFloat);
		return torch::from_blob(buf.data(), { (int64_t)rows, (int64_t)innerSize }, options).clone();
	}

	inline FList TENSOR_TO_FLIST(torch::Tensor tensor) {
		assert(tensor.dim() == 1);
		tensor = tensor.cpu().detach().to(torch::kFloat32);
		float* data = tensor.data_ptr<float>();
		return FList(data, data + tensor.size(0));
	}

	inline IList TENSOR_TO_ILIST(torch::Tensor tensor) {
		assert(tensor.dim() == 1);
		tensor = tensor.cpu().detach().to(torch::kInt32);
		int* data = tensor.data_ptr<int>();
		return IList(data, data + tensor.size(0));
	}
}