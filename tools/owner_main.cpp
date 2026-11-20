// owner_main.cpp — P4 active-set owner-map parity gate (C++ side).
//
// The per-episode team-config and sibling draws are RNG-driven and the Python (random.choices /
// random.shuffle) and C++ (std::discrete_distribution / std::mt19937) RNG algorithms cannot be made
// byte-identical, so the per-episode *draw* is not directly comparable (the plan's constraint #3 says
// gate on the deterministic transform, not the RNG). What IS comparable — and what the learner actually
// routes on — is the deterministic owner-map transform: given a sampled (n_blue, n_orange), a member
// count, a sibling/self decision, and a concrete shuffle permutation, the active set's per-slot owner
// assignment must be identical on both stacks.
//
// This mirrors the exact owner-map lines of RLGSC::TeamConfigMutator::ResetState (the blue-first slot
// order + "cycle member ids then permute" sibling logic) and the Python League.self_matchup /
// League.draw_sibling. We feed both sides the SAME permutation (in place of std::shuffle / random.shuffle)
// so the comparison is deterministic.
//
// Input (stdin), one case per line:
//   <memberCount> <sibling:0|1> <n_blue> <n_orange> <perm...>
//   perm is the length-(n_blue+n_orange) permutation applied to the cycled member ids.
// Output (stdout), one owner vector per case (space-separated ints), same order as input.
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {
	std::string line;
	while (std::getline(std::cin, line)) {
		if (line.empty())
			continue;
		std::istringstream ss(line);
		int memberCount, sibling, nBlue, nOrange;
		ss >> memberCount >> sibling >> nBlue >> nOrange;
		int n = nBlue + nOrange;

		// Mirror TeamConfigMutator: blue-first slot order (first nBlue slots blue, rest orange). The
		// owner list is built over the n slots; self-play -> every slot member 0; sibling -> cycle member
		// ids (j % memberCount) then apply the supplied permutation in place of std::shuffle.
		std::vector<int> botIds(n, 0);
		if (sibling) {
			for (int j = 0; j < n; j++)
				botIds[j] = j % memberCount;
			// Apply the permutation: out[j] = cycled[perm[j]].
			std::vector<int> cycled = botIds;
			for (int j = 0; j < n; j++) {
				int p;
				ss >> p;
				botIds[j] = cycled[p];
			}
		}

		for (int j = 0; j < n; j++)
			std::cout << (j ? " " : "") << botIds[j];
		std::cout << "\n";
	}
	return 0;
}
