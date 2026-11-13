#include "InferUnit.h"

#include "../../../private/RLGymPPO_CPP/FrameworkTorch.h"
#include <torch/torch.h>
#include <torch/csrc/api/include/torch/serialize.h>

using namespace RLGSC;
using namespace RLGPC;

RLGPC::InferUnit::InferUnit(
	OBSBuilder* obsBuilder, ActionParser* actionParser, 
	std::filesystem::path modelPath, bool isPolicy, const PerceiverConfig& perceiverConfig, bool gpu)
	: obsBuilder(obsBuilder), actionParser(actionParser), isPolicy(isPolicy) {

	RG_LOG("InferUnit():");

	RG_LOG(" > Creating policy/critic...");
	this->device = gpu ? torch::kCUDA : torch::kCPU;
	torch::Device device = this->device;

	if (isPolicy) {
		policy = new GaussianPolicy(perceiverConfig, device);
	} else {
		policy = NULL;
		EARLPerceiver earl = EARLPerceiver(
			perceiverConfig.nDims, perceiverConfig.nHeads, perceiverConfig.nLayers,
			perceiverConfig.queryFeatures, perceiverConfig.kvFeatures);
		critic = PerceiverNet(
			earl, torch::nn::Linear(torch::nn::LinearOptions(perceiverConfig.nDims, 1)),
			perceiverConfig.queryFeatures, perceiverConfig.kvFeatures,
			perceiverConfig.nTokens, perceiverConfig.chunkSize);
		critic->to(device, true);
	}
	RG_LOG(" > Loading policy/critic...");
	try {
		auto streamIn = std::ifstream(modelPath, std::ios::binary);
		torch::load(policy ? policy->net.ptr() : critic.ptr(), streamIn, device);
	} catch (std::exception& e) {
		RG_ERR_CLOSE(
			"Failed to load model, checkpoint may be corrupt or of different model arch.\n" <<
			"Exception: " << e.what()
		);
	}

	RG_LOG(" > Done!");
}

RLGPC::InferUnit::~InferUnit() {
	delete policy;
}

RLGSC::FList RLGPC::InferUnit::GetObs(const RLGSC::PlayerData& player, const RLGSC::GameState& state, const RLGSC::Action& prevAction) {
	return obsBuilder->BuildOBS(player, state, prevAction);
}

RLGSC::FList2 RLGPC::InferUnit::GetObs(const RLGSC::GameState& state, const RLGSC::ActionSet& prevActions) {
	FList2 obsSet = {};
	for (int i = 0; i < state.players.size(); i++)
		obsSet.push_back(obsBuilder->BuildOBS(state.players[i], state, prevActions[i]));
	return obsSet;
}

#define ASSERT_IS_POLICY() \
if (!isPolicy) RG_ERR_CLOSE("InferUnit: Failed to infer the policy because this inference unit was created to infer the critic");
#define ASSERT_IS_CRITIC() \
if (isPolicy) RG_ERR_CLOSE("InferUnit: Failed to infer the critic because this inference unit was created to infer the policy");

ActionSet RLGPC::InferUnit::InferPolicyAll(
	const GameState& state, const ActionSet& prevActions, 
	bool deterministic
) {

	ASSERT_IS_POLICY();

	FList2 obsSet = GetObs(state, prevActions);
	
	RG_NOGRAD;
	torch::Tensor inputTen = FLIST2_TO_TENSOR(obsSet).to(device);
	auto actionResult = policy->GetAction(inputTen, deterministic);
	auto actionParserInput = TENSOR_TO_FLIST(actionResult.action.cpu().flatten());

	return actionParser->ParseActions(actionParserInput, state);
}

Action RLGPC::InferUnit::InferPolicySingle(
	const PlayerData& player, const GameState& state, const Action& prevAction, 
	bool deterministic
) {
	ASSERT_IS_POLICY();

	FList obs = GetObs(player, state, prevAction);

	int playerIndex = 0;
	for (int i = 1; i < state.players.size(); i++) {
		if (state.players[i].carId == player.carId) {
			playerIndex = i;
			break;
		}
	}

	RG_NOGRAD;
	torch::Tensor inputTen = torch::tensor(obs).to(device);
	auto actionResult = policy->GetAction(inputTen, deterministic);
	FList actionParserInput = TENSOR_TO_FLIST(actionResult.action.cpu().flatten());

	return actionParser->ParseActions(actionParserInput, state)[playerIndex];
}

RLGSC::FList RLGPC::InferUnit::InferCriticAll(const RLGSC::GameState& state, const RLGSC::ActionSet& prevActions) {
	ASSERT_IS_CRITIC();

	FList2 obsSet = GetObs(state, prevActions);

	RG_NOGRAD;
	torch::Tensor inputTen = FLIST2_TO_TENSOR(obsSet).to(device);
	return TENSOR_TO_FLIST(critic->Forward(inputTen).cpu());
}

float RLGPC::InferUnit::InferCriticSingle(const RLGSC::PlayerData& player, const RLGSC::GameState& state, const RLGSC::Action& prevAction) {
	ASSERT_IS_CRITIC();

	FList obs = GetObs(player, state, prevAction);

	RG_NOGRAD;
	torch::Tensor inputTen = torch::tensor(obs).to(device);
	return critic->Forward(inputTen).cpu().item<float>();
}
