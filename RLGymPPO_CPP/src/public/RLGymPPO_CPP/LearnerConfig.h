#pragma once
#include "Lists.h"
#include "PPO/PPOLearnerConfig.h"
#include <RLGymPPO_CPP/Util/SkillTrackerConfig.h>

namespace RLGPC {
	enum class LearnerDeviceType {
		AUTO,
		CPU,
		GPU_CUDA
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	struct LearnerConfig {
		int numThreads = 8;
		int numGamesPerThread = 16;
		int minInferenceSize = 80;

		bool renderMode = false;
		// If renderMode, this is the scaling of time for the game
		// 1.0 = Run the game at real time
		// 2.0 = Run the game twice as fast as real time
		float renderTimeScale = 1.5f; 

		// Enable rendering during training
		// WARNING: This can cause tons of lag and another issues, and sometimes crashes due to python lagging out
		bool renderDuringTraining = false;

		// Set to 0 to disable
		uint64_t timestepLimit = 0;

		int64_t expBufferSize = 100 * 1000;
		int64_t timestepsPerIteration = 50 * 1000;

		// Match-based collection (the reference implementation's curriculum driver). When > 0, one
		// iteration collects this many COMPLETED episodes (each episode's `done`, including a drill
		// cut short by its time cap, which still emits `done`) instead of stopping on a raw step
		// target. This makes the per-epoch metric a whole-match mean (comparable across drill/native
		// episode lengths) rather than a step-budget mean. `timestepsPerIteration` then acts only as a
		// hang-guard: collection stops at whichever comes first (match target reached, or step guard
		// hit), so a pathological never-ending match can't hang the loop. 0 disables match mode and
		// falls back to the pure step target (legacy behavior).
		int64_t matchesPerIteration = 0;

		bool standardizeReturns = true;
		bool standardizeOBS = false; // TODO: Implement
		int maxReturnsPerStatsInc = 150;
		int stepsPerObsStatsInc = 5;

		// Actions with the highest probability are always chosen, instead of being more likely
		// This will make your bot play better, but is horrible for learning
		// Trying to run a PPO learn iteration with deterministic mode will throw an exception
		bool deterministic = false;

		// Collect additional steps during the learning phase
		// Note that, once the learning phase completes and the policy is updated, these additional steps are from the old policy
		bool collectionDuringLearn = false;

		PPOLearnerConfig ppo = {};

		// Multi-member league config. When memberPerceiver is non-empty, the learner builds one PPO
		// learner per member (each with its own Perceiver obs width). The shared PPO hyperparams come from
		// `ppo`; only `perceiver` varies per member. Empty memberPerceiver => single-member (uses
		// ppo.perceiver). Which member controls each slot is decided PER EPISODE by the active-set setter
		// (see EnvConfig), not by a frozen owner map.
		std::vector<PerceiverConfig> memberPerceiver = {};

		float gaeLambda = 0.95f;
		float gaeGamma = 0.99f;
		float rewardClipRange = 10; // Clip range for normalized rewards, set 0 to disable

		// Set to a directory with numbered subfolders, the learner will load the subfolder with the highest number
		// If the folder is empty or does not exist, loading is skipped
		// Set empty to disable loading entirely
		std::filesystem::path checkpointLoadFolder = "checkpoints"; 

		// Checkpoints are saved here as timestep-numbered subfolders
		//	e.g. a checkpoint at 20,000 steps will save to a subfolder called "20000"
		// Set empty to disable saving
		std::filesystem::path checkpointSaveFolder = "checkpoints"; 
		bool saveFolderAddUnixTimestamp = false; // Appends the unix time to checkpointSaveFolder

		// Save every timestep
		// Set to zero to just use timestepsPerIteration
		int64_t timestepsPerSave = 500 * 1000;

		int randomSeed = 123;
		int checkpointsToKeep = 5; // Checkpoint storage limit before old checkpoints are deleted, set to -1 to disable
		LearnerDeviceType deviceType = LearnerDeviceType::AUTO; // Auto will use your CUDA GPU if available

		// TF32 (O3a port): on Ampere+ (sm_86 = RTX 3090) the cuBLAS/cuDNN matmul+conv kernels default to
		// full FP32. Enabling TF32 lets those kernels use the 19-bit tensor-core path — the reference O3a
		// lever, measured 1.62x on the GPU step. It is a precision change (the matmul reduction accumulates
		// in TF32), so it is opt-in and gated on the same relaxed ML quality bar as autocast. No-op on CPU.
		bool useTF32 = false;

		// Folder containing the RocketSim collision meshes (soccar/hoops/dropshot subfolders of .cmf
		// files). Relative paths resolve against the process CWD, so prefer an ABSOLUTE path when the
		// learner is embedded in a host (e.g. a Python driver) whose CWD you don't control.
		std::filesystem::path collisionMeshesFolder = "collision_meshes";

		// Send metrics to the python metrics receiver
		// The receiver can then log them to wandb or whatever
		bool sendMetrics = true;
		std::string metricsProjectName = "rlgymppo-cpp"; // Project name for the python metrics receiver
		std::string metricsGroupName = "unnamed-runs"; // Group name for the python metrics receiver
		std::string metricsRunName = "rlgymppo-cpp-run"; // Run name for the python metrics receiver

		SkillTrackerConfig skillTrackerConfig = {};
	};
}