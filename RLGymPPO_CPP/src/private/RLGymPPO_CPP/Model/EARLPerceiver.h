#pragma once
#include <torch/torch.h>

// Native libtorch port of earl_pytorch.model.earl_perceiver (EARLPerceiver +
// EARLPerceiverBlock). The module/parameter NAMES mirror the Python module tree
// exactly (attention.in_proj_weight, attention.out_proj.weight, blocks.N.linear1.*,
// query_preprocess.0/2.*, ...) so the BC checkpoint state dicts load 1:1 via
// torch::load(module, stream).
//
// WHY a hand-ported MHA instead of torch::nn::MultiheadAttention:
//   The Python agent calls nn.MultiheadAttention(batch_first=True), which dispatches
//   to the PURE-PYTHON torch.nn.functional.multi_head_attention_forward. That impl
//   treats a FLOAT key_padding_mask as an ADDITIVE logit bias (baddbmm(attn_mask, ...)).
//   The C++ ATen multi_head_attention_forward (what torch::nn::MultiheadAttention calls)
//   instead requires a BOOL key_padding_mask and masked_fill_(-inf)s it. The two are
//   incompatible for the float mask the sim emits, so we replicate the Python functional
//   by hand. Parameter names are kept identical so the checkpoint still loads.
//
// Semantics mirrored from earl_pytorch (cross-attention: query=src, key=value=invariant):
//   block: src2 = norm1(src); inv = norm2(inv);
//          src2 = MHA(src2, inv, inv, key_padding_mask=mask); src = src + src2;
//          src2 = linear2(relu(linear1(norm3(src)))); src = src + src2
//   net:   q_emb = query_preprocess(q); kv_emb = key_value_preprocess(kv);
//          for each block: q_emb = block(q_emb, kv_emb, mask); return q_emb
// Preprocess MLPs: Sequential(Linear(feat,256), ReLU, Linear(256,256), ReLU)
// (state-dict indices 0 and 2). Every dim>1 parameter is xavier_uniform_(gain=1)
// at block construction (Python EARLPerceiverBlock._reset_parameters).

namespace RLGPC {
	// Hand-ported nn.MultiheadAttention (encoder-decoder / cross-attention path, since
	// the EARL block passes query != key==value). Operates batch-first: query (B,L,E),
	// key/value (B,S,E), float key_padding_mask (B,S) added to the logits.
	class MultiheadAttentionImpl : public torch::nn::Module {
	public:
		// Registered with the SAME names the checkpoint uses under attention.*:
		//   attention.in_proj_weight (3E,E), attention.in_proj_bias (3E),
		//   attention.out_proj.weight (E,E), attention.out_proj.bias (E)
		torch::Tensor in_proj_weight, in_proj_bias;
		torch::nn::Linear out_proj{nullptr};

		int64_t embedDim;
		int64_t numHeads;

		MultiheadAttentionImpl(int64_t embedDim, int64_t numHeads);

		torch::Tensor forward(const torch::Tensor& query, const torch::Tensor& key,
			const torch::Tensor& value, const torch::Tensor& keyPaddingMask);
	};

	TORCH_MODULE(MultiheadAttention);

	class EARLPerceiverBlock : public torch::nn::Module {
	public:
		MultiheadAttention attention{nullptr};
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
