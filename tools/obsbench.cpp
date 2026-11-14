// Standalone microbenchmark for the per-step observation->tensor conversion.
//
// The hot collection loop (ThreadAgent::_RunFunc) builds the model input every step via
// MakeGamesOBSTensor(): for each game it calls FLIST2_TO_TENSOR (a torch::zeros plus one
// slice/copy_ ATen dispatch PER PLAYER ROW) and then torch::concat across games. That is
// dispatch-bound on the CPU side, exactly like the original reference obs-builder before it
// was fused. This file measures the old path vs a fused single-buffer path and PROVES the
// two produce bit-identical float32 tensors.
//
// It is intentionally self-contained (own FList2 typedef, own copies of both impls) so it
// compiles against torch alone, independent of the repo headers.

#include <torch/torch.h>

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

typedef std::vector<float> FList;
typedef std::vector<std::vector<float>> FList2;

// ---- OLD (current repo impl, verbatim) -------------------------------------------------
static torch::Tensor FLIST2_TO_TENSOR_old(const FList2& list) {
	int innerSize = list[0].size();
	auto options = torch::TensorOptions().dtype(at::kFloat);
	auto tensor = torch::zeros({ (int)list.size(), innerSize }, options);
	for (int i = 0; i < list.size(); i++)
		tensor.slice(0, i, i + 1) = torch::from_blob((float*)list[i].data(), { innerSize }, options);
	return tensor;
}

// OLD MakeGamesOBSTensor: per-game tensor then concat (matches ThreadAgent.cpp).
static torch::Tensor MakeGamesOBS_old(const std::vector<FList2>& games) {
	std::vector<torch::Tensor> obsTensors;
	for (auto& g : games)
		obsTensors.push_back(FLIST2_TO_TENSOR_old(g));
	return torch::concat(obsTensors);
}

// ---- NEW (fused: one contiguous buffer, one tensor build) ------------------------------
static torch::Tensor FLIST2_TO_TENSOR_new(const FList2& list) {
	int rows = (int)list.size();
	int innerSize = list[0].size();
	std::vector<float> buf;
	buf.reserve((size_t)rows * (size_t)innerSize);
	for (auto& row : list)
		buf.insert(buf.end(), row.begin(), row.end());
	auto options = torch::TensorOptions().dtype(at::kFloat);
	return torch::from_blob(buf.data(), { (int64_t)rows, (int64_t)innerSize }, options).clone();
}

// NEW MakeGamesOBSTensor: flatten every player's obs across every game into one buffer,
// build one (totalPlayers, obsSize) tensor. No per-game tensor, no concat.
static torch::Tensor MakeGamesOBS_new(const std::vector<FList2>& games) {
	int totalPlayers = 0;
	for (auto& g : games) totalPlayers += (int)g.size();
	int obsSize = games[0][0].size();
	std::vector<float> buf;
	buf.reserve((size_t)totalPlayers * (size_t)obsSize);
	for (auto& g : games)
		for (auto& row : g)
			buf.insert(buf.end(), row.begin(), row.end());
	auto options = torch::TensorOptions().dtype(at::kFloat);
	return torch::from_blob(buf.data(), { (int64_t)totalPlayers, (int64_t)obsSize }, options).clone();
}

// ---- harness ----------------------------------------------------------------------------
static FList2 makeObs(int players, int cols, std::mt19937& rng) {
	std::uniform_real_distribution<float> dist(-1.f, 1.f);
	FList2 out(players, FList(cols));
	for (auto& row : out)
		for (auto& v : row) v = dist(rng);
	return out;
}

template <class F>
static double timeUs(int iters, F fn) {
	// warmup
	for (int i = 0; i < 50; i++) fn();
	auto t0 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iters; i++) fn();
	auto t1 = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

int main() {
	at::set_num_threads(1); // isolate single-thread dispatch cost
	std::mt19937 rng(1234);

	const int OBS = 10778; // queryFeatures + nTokens*kvFeatures + nTokens
	const int ITERS = 3000;

	// Single-game case (1v1, 2 players)
	{
		FList2 obs = makeObs(2, OBS, rng);
		auto a = FLIST2_TO_TENSOR_old(obs);
		auto b = FLIST2_TO_TENSOR_new(obs);
		bool eq = torch::equal(a, b);
		double tOld = timeUs(ITERS, [&]{ return FLIST2_TO_TENSOR_old(obs); });
		double tNew = timeUs(ITERS, [&]{ return FLIST2_TO_TENSOR_new(obs); });
		printf("[FLIST2 1v1] bit-exact=%d  old=%.3fus  new=%.3fus  speedup=%.2fx\n",
			(int)eq, tOld, tNew, tOld / tNew);
	}

	// Multi-game case (4 games x 2 players) -> the concat path
	{
		std::vector<FList2> games;
		for (int i = 0; i < 4; i++) games.push_back(makeObs(2, OBS, rng));
		auto a = MakeGamesOBS_old(games);
		auto b = MakeGamesOBS_new(games);
		bool eq = torch::equal(a, b);
		double tOld = timeUs(ITERS, [&]{ return MakeGamesOBS_old(games); });
		double tNew = timeUs(ITERS, [&]{ return MakeGamesOBS_new(games); });
		printf("[MakeGames 4x2] bit-exact=%d  old=%.3fus  new=%.3fus  speedup=%.2fx\n",
			(int)eq, tOld, tNew, tOld / tNew);
	}

	// 8-player single game (stress the per-row dispatch count)
	{
		FList2 obs = makeObs(8, OBS, rng);
		auto a = FLIST2_TO_TENSOR_old(obs);
		auto b = FLIST2_TO_TENSOR_new(obs);
		bool eq = torch::equal(a, b);
		double tOld = timeUs(ITERS, [&]{ return FLIST2_TO_TENSOR_old(obs); });
		double tNew = timeUs(ITERS, [&]{ return FLIST2_TO_TENSOR_new(obs); });
		printf("[FLIST2 8p] bit-exact=%d  old=%.3fus  new=%.3fus  speedup=%.2fx\n",
			(int)eq, tOld, tNew, tOld / tNew);
	}
	return 0;
}
