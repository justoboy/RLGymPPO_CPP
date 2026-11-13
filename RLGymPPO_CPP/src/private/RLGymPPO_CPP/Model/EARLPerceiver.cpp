#include "EARLPerceiver.h"
#include <memory>

// Exact port of earl_pytorch.model.earl_perceiver. Registered module names match the
// Python attribute names 1:1 so BC state dicts load without remapping:
//   blocks.N.attention.in_proj_weight / in_proj_bias / out_proj.weight / out_proj.bias
//   blocks.N.linear1.weight ... blocks.N.norm3.bias
//   query_preprocess.0/2.* key_value_preprocess.0/2.*

RLGPC::EARLPerceiverBlock::EARLPerceiverBlock(int nDims, int nHeads) {
	// Python: nn.MultiheadAttention(n_dims, n_heads, batch_first=True). The C++ module has no
	// batch_first option, so forward() transposes (B,S,E)->(S,B,E) manually; the ATen op
	// underneath is the same implementation either way. average_attn_weights defaults true on
	// both stacks, and need_weights=true matches the Python call (weights returned, unused).
	attention = register_module("attention", torch::nn::MultiheadAttention(
		torch::nn::MultiheadAttentionOptions(nDims, nHeads)));
	// Python: linear1 = Linear((1+concatenate)*n_dims, 4*n_dims); concatenate=False
	linear1 = register_module("linear1", torch::nn::Linear(torch::nn::LinearOptions(nDims, 4 * nDims)));
	linear2 = register_module("linear2", torch::nn::Linear(torch::nn::LinearOptions(4 * nDims, nDims)));
	norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ (int64_t)nDims })));
	norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ (int64_t)nDims })));
	norm3 = register_module("norm3", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ (int64_t)nDims })));

	// Python EARLPerceiverBlock._reset_parameters: xavier_uniform_(p) on every dim>1 parameter
	// (PyTorch default gain = 1.0).
	for (auto& p : named_parameters()) {
		if (p.value().dim() > 1)
			torch::nn::init::xavier_uniform_(p.value());
	}
}

torch::Tensor RLGPC::EARLPerceiverBlock::forward(
	const torch::Tensor& src, const torch::Tensor& invariant, const torch::Tensor& mask) {
	// Python: src2 = norm1(src); invariant = norm2(invariant);
	//         src2, _ = attention(src2, invariant, invariant, key_padding_mask=mask); src += src2
	// C++ MHA is seq-first: transpose (B,S,E)->(S,B,E), attend, transpose back. The float
	// key_padding_mask is passed through _canonical_mask unchanged and added to the attention
	// logits by the same ATen op on both stacks (parity by construction).
	auto q = norm1->forward(src).transpose(0, 1);
	auto inv = norm2->forward(invariant).transpose(0, 1);
	auto attn = attention->forward(q, inv, inv, mask, true, torch::Tensor(), true);
	auto x = src + std::get<0>(attn).transpose(0, 1);

	// Python: src2 = norm3(src); src2 = linear2(relu(linear1(src2))); src = src + src2
	auto normed = norm3->forward(x);
	auto mlpOut = linear2->forward(torch::relu(linear1->forward(normed)));
	return x + mlpOut;
}

RLGPC::EARLPerceiverImpl::EARLPerceiverImpl(
	int nDims, int nHeads, int nLayers, int queryFeatures, int keyValueFeatures) : nDims(nDims) {
	// Python mlp(in, hidden, hidden_layers=1) == Sequential(Linear, ReLU, Linear, ReLU)
	// (state-dict indices 0 and 2 are the Linears).
	query_preprocess = register_module("query_preprocess", torch::nn::Sequential(
		torch::nn::Linear(torch::nn::LinearOptions(queryFeatures, nDims)), torch::nn::ReLU(),
		torch::nn::Linear(torch::nn::LinearOptions(nDims, nDims)), torch::nn::ReLU()));
	key_value_preprocess = register_module("key_value_preprocess", torch::nn::Sequential(
		torch::nn::Linear(torch::nn::LinearOptions(keyValueFeatures, nDims)), torch::nn::ReLU(),
		torch::nn::Linear(torch::nn::LinearOptions(nDims, nDims)), torch::nn::ReLU()));

	blocks = register_module("blocks", torch::nn::ModuleList());
	for (int i = 0; i < nLayers; i++)
		blocks->push_back(EARLPerceiverBlock(nDims, nHeads));
}

torch::Tensor RLGPC::EARLPerceiverImpl::forward(
	const torch::Tensor& queryEntities, const torch::Tensor& keyValueEntities,
	const torch::Tensor& mask) {
	auto qEmb = query_preprocess->forward(queryEntities);
	auto kvEmb = key_value_preprocess->forward(keyValueEntities);

	// Python: for block in self.blocks: q_emb, _ = block(q_emb, kv_emb, mask=mask)
	for (auto& child : blocks->children())
		qEmb = std::static_pointer_cast<EARLPerceiverBlock>(child)->forward(qEmb, kvEmb, mask);

	return qEmb; // postprocess is Identity (n_postprocess_layers=0)
}
