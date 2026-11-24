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

#include <string>
#include <utility>
#include <vector>

namespace RLGPC {

	// One league member's reward "program": the set of terms that bot selects (name -> weight) plus its
	// own team_spirit / opponent_punish scalars. Mirrors the per-slot tuple in the reference reward layer.
	struct MemberRewardConfig {
		std::vector<std::pair<std::string, float>> termWeights; // selected term name -> weight
		float teamSpirit = 0.6f;
		float opponentPunish = 1.0f;
	};

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

		// Per-episode active-set / matchup sampler (P4). When teamConfigs is non-empty, the env uses the
		// TeamConfigMutator: on every reset it samples one (n_blue, n_orange) from the weighted pool,
		// clears the arena, respawns exactly that active set, runs a native random kickoff, and stamps the
		// per-episode slot->owner map (which member controls each slot) into GameState::slotOwners. When
		// empty, the env uses the plain random-kickoff setter (single-bot self-play, every slot -> member 0).
		std::vector<std::pair<int, int>> teamConfigs = {};
		std::vector<float> teamWeights = {};
		int memberCount = 1;
		float siblingFraction = 0.0f;
		int randomSeed = 123; // per-env sampler seed (each game's sampler draws its own episodes)

		// Two-wheel spawn curriculum (touch + goal). When wheelEnabled, the TeamConfigMutator may override a
		// fraction of episodes' native kickoff with a drill spawn (touch: car at distance d from a center ball;
		// goal: a 1v0 breakaway with the ball on the lone car's attack axis). The driver advances each wheel's
		// streak/score state machine per epoch and writes the per-wheel geometry + fire-probability via
		// SetWheelConfig(); wheelEnabled just gates whether the curriculum is active at all. Disabled =>
		// every episode uses the native random kickoff.
		bool  wheelEnabled    = false;
		float wheelDrillTime  = 10.0f;   // clock cap (s) on a drill episode
		float wheelNativeTime = 300.0f;  // clock cap (s) on a native (non-drill) episode

		// Overtime grace (s) before a tied game is force-truncated (the ScoreboardTimeCondition terminal).
		float maxOvertimeSeconds = 300.0f;

		// Per-member reward spec (one per league member, indexed by owner id == slotOwners value). When
		// non-empty, the env uses the per-bot ComposedReward (term routing + team_spirit/opponent_punish
		// blend at the explicit n_blue boundary) instead of the generic CombinedReward. Empty => CombinedReward.
		std::vector<MemberRewardConfig> memberRewardConfigs = {};

		// Fixed arena mode for this env (the league builds one env per mode). Maps to GameMode.
		// The arena's mode is fixed at Arena::Create() (Gym construction) time, so mode selection is
		// per-env, not per-episode; the single StateSetter is just the random-kickoff resetter.
		GameMode gameMode = GameMode::SOCCAR;

		// Perceiver obs layout must match TemporalObsBuilder's output for this config.
		int queryFeatures = 208;
		int kvFeatures = 69;
		int nTokens = 151;
		int nControls = 9;
		int entityFeatures = 39;

		// Actor head type (the switchable-arm knob). Selects the env's action parser: CONTINUOUS ->
		// ContinuousActionParser (reads the 9 floats as control values); DISCRETE ->
		// PerChannelDiscreteActionParser (decodes the 9 bin-indices via the fixed per-channel centers).
		// Must match the perceiver's headType so the policy's action encoding and the env's decode agree.
		HeadType headType = HeadType::CONTINUOUS;
	};

	// Factory that builds a Learner from an EnvConfig + LearnerConfig. The single generic verb the
	// Python driver uses to spin up one member's learner. Defined in the DLL (EnvFactory.cpp).
	RG_IMEXPORT Learner* CreateLearner(const EnvConfig& ec, const LearnerConfig& cfg);
}
