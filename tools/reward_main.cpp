// reward_main.cpp — GAP1/2/3 reward route+blend parity gate (C++ side).
//
// The C++ learner's ComposedReward (RLGymSim_CPP/.../RewardFunctions/RewardTerms.h) does two things the
// old generic CombinedReward could not: (1) per-BOT term routing — each slot's contribution is the weighted
// sum of ONLY the terms its OWNER selects, and (2) team_spirit / opponent_punish blending at the EXPLICIT
// blue-count boundary (so uneven teams split correctly). The raw term values themselves come from physics
// (identical inputs on both stacks), so the only thing that can diverge is the deterministic route+blend
// transform. This gate locks exactly that transform.
//
// It re-implements, on plain float vectors, the same math as ComposedReward::GetAllRewards:
//   routed[j] = sum over terms i of weight[i][owner[j]] * contrib[i][j]
//     where contrib[i][j] = isDelta[i] ? (raw[i][j] - prev[i][j]) : raw[i][j]
//   bm = mean(routed[0:nBlue]), om = mean(routed[nBlue:n])
//   out[j] = (1-ts[owner[j]])*routed[j] + ts[owner[j]]*teamMean(j) - op[owner[j]]*oppMean(j)
//     where for a blue slot teamMean=bm, oppMean=om; for an orange slot teamMean=om, oppMean=bm.
//
// The Python mirror (reward_fixture.py) implements the identical transform and feeds BOTH the same synthetic
// per-term raw vectors / owner map / weights / ts / op. If they agree on every case, the C++ reward routing
// matches the reference reward semantics.
//
// Input (stdin), one case per record. A record is:
//   <n> <nBlue> <memberCount> <numTerms>
//   <owner[0..n-1]>                         (n ints)
//   <teamSpirit[0..memberCount-1]>          (memberCount floats)
//   <opponentPunish[0..memberCount-1]>      (memberCount floats)
//   then for each term i in 0..numTerms-1:
//     <isDelta:0|1> <weight[0..memberCount-1]> <raw[0..n-1]> <prev[0..n-1]>
//       isDelta (1 int), then memberCount weights, then n raw floats, then n prev floats
// Output (stdout): one blended per-slot reward vector per case (space-separated floats), same order.
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {
	std::string line;
	while (true) {
		// Header line: n nBlue memberCount numTerms
		std::string header;
		if (!std::getline(std::cin, header))
			break;
		// Skip blank separator lines between records (empty or whitespace-only).
		while (header.find_first_not_of(" \t\r") == std::string::npos) {
			if (!std::getline(std::cin, header))
				return 0;
		}
		std::istringstream hs(header);
		int n, nBlue, memberCount, numTerms;
		if (!(hs >> n >> nBlue >> memberCount >> numTerms))
			break;

		// owner[j]
		std::vector<int> owner(n);
		{
			std::string l; std::getline(std::cin, l);
			std::istringstream ss(l);
			for (int j = 0; j < n; j++) ss >> owner[j];
		}
		// teamSpirit[m], opponentPunish[m]
		std::vector<float> ts(memberCount), op(memberCount);
		{
			std::string l; std::getline(std::cin, l);
			std::istringstream ss(l);
			for (int m = 0; m < memberCount; m++) ss >> ts[m];
		}
		{
			std::string l; std::getline(std::cin, l);
			std::istringstream ss(l);
			for (int m = 0; m < memberCount; m++) ss >> op[m];
		}
		// Per-term: isDelta, weights[memberCount], raw[n], prev[n]
		std::vector<int> isDelta(numTerms);
		std::vector<std::vector<float>> weight(numTerms, std::vector<float>(memberCount, 0.0f));
		std::vector<std::vector<float>> raw(numTerms, std::vector<float>(n, 0.0f));
		std::vector<std::vector<float>> prev(numTerms, std::vector<float>(n, 0.0f));
		for (int i = 0; i < numTerms; i++) {
			std::string l; std::getline(std::cin, l);
			std::istringstream ss(l);
			int d; ss >> d; isDelta[i] = d;
			for (int m = 0; m < memberCount; m++) ss >> weight[i][m];
			for (int j = 0; j < n; j++) ss >> raw[i][j];
			for (int j = 0; j < n; j++) ss >> prev[i][j];
		}

		// 1. Route: per-slot weighted sum over terms the owner selects (weight != 0).
		std::vector<float> routed(n, 0.0f);
		for (int i = 0; i < numTerms; i++) {
			for (int j = 0; j < n; j++) {
				float contrib = isDelta[i] ? (raw[i][j] - prev[i][j]) : raw[i][j];
				int o = owner[j];
				if (o < 0 || o >= memberCount) o = 0;
				float w = weight[i][o];
				if (w != 0.0f)
					routed[j] += w * contrib;
			}
		}

		// 2. Blend at the explicit nBlue boundary (blue-first slot order).
		float bm = 0.0f, om = 0.0f; int nb = 0, no = 0;
		for (int j = 0; j < n; j++) {
			if (j < nBlue) { bm += routed[j]; nb++; }
			else         { om += routed[j]; no++; }
		}
		if (nb) bm /= (float)nb;
		if (no) om /= (float)no;

		std::vector<float> out(n, 0.0f);
		for (int j = 0; j < n; j++) {
			int o = owner[j];
			if (o < 0 || o >= memberCount) o = 0;
			float own = routed[j];
			if (j < nBlue) out[j] = (1.0f - ts[o]) * own + ts[o] * bm - op[o] * om;
			else           out[j] = (1.0f - ts[o]) * own + ts[o] * om - op[o] * bm;
		}

		for (int j = 0; j < n; j++) {
			char buf[64];
			snprintf(buf, sizeof(buf), "%.9g", out[j]);
			std::cout << (j ? " " : "") << buf;
		}
		std::cout << "\n";
	}
	return 0;
}
