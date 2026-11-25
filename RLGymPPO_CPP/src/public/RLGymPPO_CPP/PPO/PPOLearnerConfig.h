#pragma once
#include "../Lists.h"
#include <vector>

namespace RLGPC {
	// Which actor head the PerceiverNet trunk projects into. Mirrors the reference
	// BotSpec.head ("continuous" | "discrete"): the SAME EARLPerceiver trunk feeds either
	// a continuous Gaussian head or a per-channel categorical head. The head type is the
	// single source of truth for how the actor's output is interpreted (and how the env
	// action parser decodes it), so BC training, RL rollout, and live inference all agree.
	enum class HeadType { CONTINUOUS, DISCRETE };

	// Architecture of the EARL Perceiver actor/critic. The obs layout describes how the
	// FLAT per-player obs (built by the sim's TemporalObsBuilder) is sliced back into the
	// (q, kv, mask, mode) tuple the Perceiver consumes:
	//   flat obs = q(queryFeatures) ++ kv(nTokens * kvFeatures) ++ m(nTokens) ++ mode(1)
	// Defaults match the reference main BotSpec (profile=True): entity_features=39,
	// query_features=195, kv_features=64, 151 temporal slots (13 frames x 9 ball+cars + 34 pads).
	struct PerceiverConfig {
		// NOTE: matches the reference _make_earl == EARLPerceiver(256, 4, 8, 1, ...),
		// i.e. n_layers=4, n_heads=8 (the plan text had these swapped; the shipped BC
		// checkpoints have 4 blocks, confirmed by their state-dict keys).
		int nDims = 256;   // latent/embedding width
		int nHeads = 8;    // attention heads (256/8 = 32 per head)
		int nLayers = 4;   // cross-attention blocks
		int nControls = 9; // continuous control channels (throttle..release)

		// Actor head selection (the switchable-arm knob). CONTINUOUS -> Gaussian head
		// (9 means + 9 log_stds); DISCRETE -> per-channel categorical over nBins.
		HeadType headType = HeadType::CONTINUOUS;

		// Per-channel bin counts for the DISCRETE arm. Mirrors the reference N_BINS:
		//   col0 throttle(analog) col1 steer(analog) col2 pitch(ternary) col3 yaw(analog)
		//   col4 roll(ternary) col5..8 binary -> [9,9,3,9,3,2,2,2,2]
		std::vector<int> nBins = { 9, 9, 3, 9, 3, 2, 2, 2, 2 };

		// Obs tuple layout (must match the sim's TemporalObsBuilder output)
		int queryFeatures = 195; // q_self(34, non-IS_* cols) + ball(6) + tactical(3) + mode-fields(10) + tiles(140) + team-size(2)
		int kvFeatures = 64;     // entity features (39) + relative block (25)
		int nTokens = 151;       // temporal slots (13 frames x 9 ball+cars + 34 boost pads)

		// FSP auxiliary head: the self row's full feature width (39 when profile on). The FSP head's
		// output width is entityFeatures - 5 (the q_self, i.e. the self row's non-IS_* cols), which is
		// also the FSP target (the first entityFeatures-5 columns of the next obs).
		int entityFeatures = 39;

		int64_t chunkSize = 2048; // PerceiverNet forward chunking (VRAM bound)
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
	struct PPOLearnerConfig {
		PerceiverConfig perceiver = {};

		int64_t batchSize = 50 * 1000;
		int epochs = 10;
		float policyLR = 1e-4f; // Policy learning rate (reference BC used AdamW 1e-4)
		float criticLR = 1e-4f; // Critic learning rate
		float entCoef = 0.005f; // Entropy coefficient
		float clipRange = 0.2f;
		int64_t miniBatchSize = 0; // Set to 0 to just use batchSize

		// FSP auxiliary-loss coefficient (reference run_ppo_update lambda_fsp). 0 disables the FSP
		// head entirely (no head built, no FSP loss term) so single-bot BC->RL stays byte-identical.
		float lambdaFSP = 1.0f;

		// Experimental, improves PPO learn speed
		// If this causes your learning to collapse, please let me know
		bool autocastLearn = false;

		// Very experimental, uses half-precision versions of models where beneficial
		bool halfPrecModels = false;

		// https://openai.com/index/how-ai-training-scales/
		// Measures the noise of both policy and critic gradients every epoch
		bool measureGradientNoise = false;
		int gradientNoiseUpdateInterval = 10;
		float gradientNoiseAvgDecay = 0.9925f;
	};
}
