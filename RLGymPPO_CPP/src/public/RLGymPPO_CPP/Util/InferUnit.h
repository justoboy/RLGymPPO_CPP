#pragma once
#include "../Lists.h"
#include "../Threading/GameInst.h"
#include "../LearnerConfig.h"

// Model types are part of this unit's public surface (the critic member is a PerceiverNet),
// so the module headers come in via relative include paths.
#include "../../../private/RLGymPPO_CPP/PPO/GaussianPolicy.h"
#include <torch/torch.h>

namespace RLGPC {
	// Standalone inference wrapper around the EARL Perceiver actor/critic. Loads a model
	// saved by PPOLearner::SaveTo (torch::save of the nn::Module) and infers actions/values.
	class RG_IMEXPORT InferUnit {
	public:

		RLGSC::OBSBuilder* obsBuilder;
		RLGSC::ActionParser* actionParser;
		class GaussianPolicy* policy;
		PerceiverNet critic{nullptr};
		bool isPolicy;
		torch::Device device{torch::kCPU};

		InferUnit(
			RLGSC::OBSBuilder* obsBuilder, RLGSC::ActionParser* actionParser, 
			std::filesystem::path modelPath, bool isPolicy, const PerceiverConfig& perceiverConfig, bool gpu = false);

		RLGSC::FList GetObs(const RLGSC::PlayerData& player, const RLGSC::GameState& state, const RLGSC::Action& prevAction);
		RLGSC::FList2 GetObs(const RLGSC::GameState& state, const RLGSC::ActionSet& prevActions);

		RLGSC::ActionSet InferPolicyAll(
			const RLGSC::GameState& state, const RLGSC::ActionSet& prevActions, 
			bool deterministic
		);
		RLGSC::Action InferPolicySingle(
			const RLGSC::PlayerData& player, const RLGSC::GameState& state, const RLGSC::Action& prevAction, 
			bool deterministic
		);

		RLGSC::FList InferCriticAll(
			const RLGSC::GameState& state, const RLGSC::ActionSet& prevActions
		);
		float InferCriticSingle(
			const RLGSC::PlayerData& player, const RLGSC::GameState& state, const RLGSC::Action& prevAction
		);

		~InferUnit();
	};
}