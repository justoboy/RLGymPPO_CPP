#include "EARLPerceiver.h"
#include <memory>
#include <cmath>

// Exact port of earl_pytorch.model.earl_perceiver + the pure-Python
// torch.nn.functional.multi_head_attention_forward (encoder-decoder / packed path).
// Registered module/parameter names match the Python attribute names 1:1 so BC state
// dicts load without remapping:
//   blocks.N.attention.in_proj_weight / in_proj_bias / out_proj.weight / out_proj.bias
//   blocks.N.linear1.weight ... blocks.N.norm3.bias
//   query_preprocess.0/2.* key_value_preprocess.0/2.*

// ---- MultiheadAttention (hand-ported functional, batch-first) -------------------

RLGPC::MultiheadAttentionImpl::MultiheadAttentionImpl(int64_t embedDim, int64_t numHeads)
	: embedDim(embedDim), numHeads(numHeads) {
	// nn.MultiheadAttention registers in_proj_weight (3E,E) and in_proj_bias (3E) as
	// plain parameters and out_proj as a Linear submodule.
	in_proj_weight = register_parameter("in_proj_weight",
		torch::empty({3 * embedDim, embedDim}, torch::TensorOptions().dtype(torch::kFloat)));
	in_proj_bias = register_parameter("in_proj_bias",
		torch::empty({3 * embedDim}, torch::TensorOptions().dtype(torch::kFloat)));
	out_proj = register_module("out_proj",
		torch::nn::Linear(torch::nn::LinearOptions(embedDim, embedDim)));
}

torch::Tensor RLGPC::MultiheadAttentionImpl::forward(const torch::Tensor& query,
	const torch::Tensor& key, const torch::Tensor& value, const torch::Tensor& keyPaddingMask) {
	// nn.MultiheadAttention(batch_first=True) transposes (B,L,E)->(L,B,E) before the
	// functional; we mirror that here and transpose back at the end.
	auto q = query.transpose(0, 1);   // (L,B,E)
	auto k = key.transpose(0, 1);     // (S,B,E)
	auto v = value.transpose(0, 1);   // (S,B,E)

	const int64_t tgtLen = q.size(0);
	const int64_t bsz = q.size(1);
	const int64_t embed = q.size(2);
	const int64_t srcLen = k.size(0);
	const int64_t headDim = embed / numHeads;

	// _in_projection_packed, encoder-decoder branch (k is v, q is not k):
	//   w_q, w_kv = w.split([E, 2E]); b_q, b_kv = b.split([E, 2E])
	const auto wQ = in_proj_weight.narrow(0, 0, embed);
	const auto wKV = in_proj_weight.narrow(0, embed, 2 * embed);
	const auto bQ = in_proj_bias.narrow(0, 0, embed);
	const auto bKV = in_proj_bias.narrow(0, embed, 2 * embed);

	auto qProj = torch::nn::functional::linear(q, wQ, bQ);     // (L,B,E)
	auto kvProj = torch::nn::functional::linear(k, wKV, bKV);  // (S,B,E)
	// unflatten(-1,(2,E)).unsqueeze(0).transpose(0,-2).squeeze(-2).contiguous()
	kvProj = kvProj.unflatten(-1, { 2, embed }).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous();
	auto kProj = kvProj.select(0, 0);  // (S,B,E)
	auto vProj = kvProj.select(0, 1);  // (S,B,E)

	// reshape for multihead, batch-first over heads: (bsz*num_heads, seq, head_dim)
	auto q2 = qProj.view({ tgtLen, bsz * numHeads, headDim }).transpose(0, 1);
	auto k2 = kProj.view({ srcLen, bsz * numHeads, headDim }).transpose(0, 1);
	auto v2 = vProj.view({ srcLen, bsz * numHeads, headDim }).transpose(0, 1);

	// merge key_padding_mask (float, additive) into attn_mask: (bsz*num_heads,1,src_len)
	auto km = keyPaddingMask.view({ bsz, 1, 1, srcLen })
		.expand({ bsz, numHeads, tgtLen, srcLen })
		.reshape({ bsz * numHeads, 1, srcLen });

	// need_weights path: scale by 1/sqrt(head_dim), baddbmm, softmax, bmm
	const double scale = std::sqrt(1.0 / static_cast<double>(headDim));
	auto qScaled = q2 * scale;
	auto attnW = torch::baddbmm(km, qScaled, k2.transpose(-2, -1));
	attnW = torch::softmax(attnW, -1);
	auto attnOut = torch::bmm(attnW, v2);  // (bsz*num_heads, tgt_len, head_dim)
	attnOut = attnOut.transpose(0, 1).reshape({ tgtLen * bsz, embed });
	attnOut = torch::nn::functional::linear(attnOut, out_proj->weight, out_proj->bias);
	attnOut = attnOut.view({ tgtLen, bsz, embed });

	// batch_first=True: transpose back to (B,L,E)
	return attnOut.transpose(0, 1);
}

// ---- EARLPerceiverBlock ---------------------------------------------------------

RLGPC::EARLPerceiverBlock::EARLPerceiverBlock(int nDims, int nHeads) {
	attention = register_module("attention", MultiheadAttention(nDims, nHeads));
	// Python: linear1 = Linear((1+concatenate)*n_dims, 4*n_dims); concatenate=False
	linear1 = register_module("linear1", torch::nn::Linear(torch::nn::LinearOptions(nDims, 4 * nDims)));
	linear2 = register_module("linear2", torch::nn::Linear(torch::nn::LinearOptions(4 * nDims, nDims)));
	norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ (int64_t)nDims })));
	norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ (int64_t)nDims })));
	norm3 = register_module("norm3", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ (int64_t)nDims })));

	// Python EARLPerceiverBlock._reset_parameters: xavier_uniform_(p) on every dim>1
	// parameter (PyTorch default gain = 1.0).
	for (auto& p : named_parameters()) {
		if (p.value().dim() > 1)
			torch::nn::init::xavier_uniform_(p.value());
	}
}

torch::Tensor RLGPC::EARLPerceiverBlock::forward(
	const torch::Tensor& src, const torch::Tensor& invariant, const torch::Tensor& mask) {
	// Python: src2 = norm1(src); invariant = norm2(invariant);
	//         src2, _ = attention(src2, invariant, invariant, key_padding_mask=mask); src += src2
	auto q = norm1->forward(src);
	auto inv = norm2->forward(invariant);
	auto attn = attention->forward(q, inv, inv, mask);
	auto x = src + attn;

	// Python: src2 = norm3(src); src2 = linear2(relu(linear1(src2))); src = src + src2
	auto normed = norm3->forward(x);
	auto mlpOut = linear2->forward(torch::relu(linear1->forward(normed)));
	return x + mlpOut;
}

// ---- EARLPerceiver --------------------------------------------------------------

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
