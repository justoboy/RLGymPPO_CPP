#pragma once

// Public, DLL-owned env factory.
//
// WHY THIS LIVES IN THE LIBRARY (not the pybind module): RLGymSim_CPP is a STATIC library that is
// linked into BOTH RLGymPPO_CPP.dll and the rlgym_learner extension module. That means RocketSim's
// global stage/mesh state exists as TWO separate copies (one per module). RocketSim::Init() is called
// by Learner (inside the DLL) and initializes the DLL's copy. If the env-creation lambda that calls
// `new Gym(...)` were compiled into the .pyd, it would read the .pyd's *uninitialized* copy of that
// static state and crash. By defining the factory HERE, the lambda's body is compiled into the DLL,
// so `new Gym` resolves to the same (initialized) RLGymSim_CPP copy that RocketSim::Init set up.
//
// The extension module only sees the *declaration* of CreateLearner (dllimport) and the plain-data
// EnvConfig struct; it never compiles any RLGymSim_CPP construction itself.

#include "Learner.h"

#include <RLGymSim_CPP/Envs/Match.h>
#include <RLGymSim_CPP/Gym.h>

namespace RLGPC {

	// Everything the C++ env factory needs to build a Match+Gym, parameterized so Python configures it
	// without touching the heavy sim types. Mirrors examplemain.cpp's EnvCreateFunc, extended with the
	// P4 mode/team/spawn-curriculum knobs.
	struct RG_IMEXPORT EnvConfig {
		int tickSkip = 8;
		float noTouchTimeoutSecs = 10.f;

		// Reward weights (CombinedReward). Mirrors the example's { FaceBall, VelPlayerToBall,
		// VelBallToGoal, EventReward{teamGoal,concede} }.
		float faceBallWeight = 0.1f;
		float velPlayerToBallWeight = 0.5f;
		float velBallToGoalWeight = 1.0f;
		float eventTeamGoal = 1.f;
		float eventConcede = -1.f;
		float eventWeight = 50.f;

		int teamSize = 1;
		bool spawnOpponents = true;

		// Fixed arena mode for this env (the league builds one env per mode). Maps to GameMode.
		GameMode gameMode = GameMode::SOCCAR;

		// Spawn-curriculum overlay (SpawnCurriculumMutator). Disabled by default -> native kickoff only.
		bool spawnCurriculumEnabled = false;
		float spawnCurriculumRatio = 1.0f;
		float spawnCurriculumProb = 0.5f;

		// Per-episode RNG seed for the kickoff + custom-spawn coin flip.
		uint32_t mutatorSeed = 0;

		// Perceiver obs layout must match TemporalObsBuilder's output for this config.
		int queryFeatures = 208;
		int kvFeatures = 69;
		int nTokens = 151;
		int nControls = 9;
		int entityFeatures = 39;
	};

	// Factory that builds a Learner from an EnvConfig + LearnerConfig. The single generic verb the
	// Python driver uses to spin up one member's learner. Defined in the DLL (EnvFactory.cpp).
	RG_IMEXPORT Learner* CreateLearner(const EnvConfig& ec, const LearnerConfig& cfg);
}
