#pragma once
#include <torch/torch.h>

// Native libtorch port of earl_pytorch.model.earl_perceiver (EARLPerceiver +
// EARLPerceiverBlock). The module/parameter NAMES mirror the Python module tree
// exactly (attention.in_proj_weight, blocks.N.linear1.weight, ...) so the BC
// checkpoint state dicts load 1:1 via torch::load(module, stream).
//
// Semantics mirrored from earl_pytorch:
//   block: src2 = norm1(src); inv = norm2(inv);
//          src2 = MHA(src2, inv, inv, key_padding_mask=mask); src = src + src2;
//          src2 = linear2(relu(linear1(norm3(src)))); src = src + src2
//   net:   q_emb = query_preprocess(q); kv_emb = key_value_preprocess(kv);
//          for each block: q_emb = block(q_emb, kv_emb, mask); return q_emb
// Preprocess MLPs: Sequential(Linear(feat,256), ReLU, Linear(256,256), ReLU)
// (state-dict indices 0 and 2). Every dim>1 parameter is xavier_uniform_(gain=1)
// at block construction (Python EARLPerceiverBlock._reset_parameters).

namespace RLGPC {
	class EARLPerceiverBlock : public torch::nn::Module {
	public:
		// Python nn.MultiheadAttention(n_dims, n_heads, batch_first=True). The C++
		// MultiheadAttention has no batch_first, so forward() transposes (B,S,E)->(S,B,E)
		// manually; the ATen op underneath is the same implementation either way.
		torch::nn::MultiheadAttention attention{nullptr};
		torch::nn::Linear linear1{nullptr}, linear2{nullptr};
		torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr}, norm3{nullptr};

		EARLPerceiverBlock(int nDims, int nHeads);

		torch::Tensor forward(
			const torch::Tensor& src, const torch::Tensor& invariant, const torch::Tensor& mask);
	};

	class EARLPerceiverImpl : public torch::nn::Module {
	public:
		torch::nn::Sequential query_preprocess{nullptr}, key_value_preprocess{nullptr};
		torch::nn::ModuleList blocks{nullptr};

		int nDims;

		EARLPerceiverImpl(int nDims, int nHeads, int nLayers, int queryFeatures, int keyValueFeatures);

		torch::Tensor forward(
			const torch::Tensor& queryEntities, const torch::Tensor& keyValueEntities,
			const torch::Tensor& mask);
	};

	TORCH_MODULE(EARLPerceiver);
}
