#pragma once
#include <RLGymPPO_CPP/Lists.h>
#include "../FrameworkTorch.h"

namespace RLGPC {

	struct ExperienceTensors {
		torch::Tensor
			states, actions, logProbs, rewards, 

#ifdef RG_PARANOID_MODE
			debugCounters,
#endif

			nextStates, dones, truncated, values, advantages,

			// FSP auxiliary: per-step target (entityFeatures-5 wide = the next obs's q_self, the self
			// row's non-IS_* cols) + mask (1 when the step is NOT an episode end, else 0). Appended
			// last so the positional begin()/end() iteration in SubmitExperience/_GetSamples carries them.
			fspTarget, fspMask;

		torch::Tensor* begin() { return &states; }
		torch::Tensor* end() { return &fspMask + 1; }
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/experience_buffer.py
	class ExperienceBuffer {
	public:

		torch::Device device;
		int seed;

		ExperienceTensors data;

		int64_t curSize = 0;
		int64_t maxSize;

		std::default_random_engine rng;

		ExperienceBuffer(int64_t maxSize, int seed, torch::Device device);

		void SubmitExperience(ExperienceTensors& data);

		struct SampleSet {
			torch::Tensor actions, logProbs, states, values, advantages, fspTarget, fspMask;
		};
		SampleSet _GetSamples(const int64_t* indices, size_t size) const;

		// Not const because it uses our random engine
		std::vector<SampleSet> GetAllBatchesShuffled(int64_t batchSize);

		void Clear();

		

		// Combine two tensors into one, removing older data if needed to fit target size
		static torch::Tensor _Concat(torch::Tensor t1, torch::Tensor t2, int64_t size);
	};
}