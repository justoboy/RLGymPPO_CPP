#include <RLGymPPO_CPP/Learner.h>

#include <RLGymSim_CPP/Utils/RewardFunctions/CommonRewards.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CombinedReward.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/NoTouchCondition.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>
#include <RLGymSim_CPP/Utils/OBSBuilders/TemporalObsBuilder.h>
#include <RLGymSim_CPP/Utils/StateSetters/RandomState.h>
#include <RLGymSim_CPP/Utils/ActionParsers/ContinuousActionParser.h>

#include "RLBotClient.h"

using namespace RLGPC; // RLGymPPO
using namespace RLGSC; // RLGymSim

// This is our step callback, it's called every step from every RocketSim game
// WARNING: This is called from multiple threads, often simultaneously, 
//	so don't access things apart from these arguments unless you know what you're doing.
// gameMetrics: The metrics for this specific game
void OnStep(GameInst* gameInst, const RLGSC::Gym::StepResult& stepResult, Report& gameMetrics) {

	auto& gameState = stepResult.state;
	for (auto& player : gameState.players) {
		// Track average player speed
		float speed = player.phys.vel.Length();
		gameMetrics.AccumAvg("player_speed", speed);

		// Track ball touch ratio
		gameMetrics.AccumAvg("ball_touch_ratio", player.ballTouchedStep);

		// Track in-air ratio
		gameMetrics.AccumAvg("in_air_ratio", !player.carState.isOnGround);
	}
}

// This is our iteration callback, it's called every time we complete an iteration, after learning
// Here we can add custom metrics to the metrics report, for example
void OnIteration(Learner* learner, Report& allMetrics) {

	AvgTracker avgPlayerSpeed = {};
	AvgTracker avgBallTouchRatio = {};
	AvgTracker avgAirRatio = {};
	
	// Get metrics for every gameInst
	auto allGameMetrics = learner->GetAllGameMetrics();
	for (auto& gameReport : allGameMetrics) {
		avgPlayerSpeed += gameReport.GetAvg("player_speed");
		avgBallTouchRatio += gameReport.GetAvg("ball_touch_ratio");
		avgAirRatio += gameReport.GetAvg("in_air_ratio");
	}

	allMetrics["player_speed"] = avgPlayerSpeed.Get();
	allMetrics["ball_touch_ratio"] = avgBallTouchRatio.Get();
	allMetrics["in_air_ratio"] = avgAirRatio.Get();
}

// Create the RLGymSim environment for each of our games
EnvCreateResult EnvCreateFunc() {
	constexpr int TICK_SKIP = 8;
	constexpr float NO_TOUCH_TIMEOUT_SECS = 10.f;

	auto rewards = new CombinedReward( // Format is { RewardFunc(), weight }
		{ 
			// Small reward for facing the ball
			{ new FaceBallReward(), 0.1f }, 

			// Moderate reward for going towards the ball
			{ new VelocityPlayerToBallReward(), 0.5f }, 

			// Bigger reward for having the ball go towards the goal
			{ new VelocityBallToGoalReward(), 1.0f }, 

			// Giant reward for scoring, giant penalty for being scored on
			{ new EventReward({.teamGoal = 1.f, .concede = -1.f}), 50.f },
		}
	);

	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(NO_TOUCH_TIMEOUT_SECS * 120 / TICK_SKIP),
		new GoalScoreCondition()
	};

	auto obs = new TemporalObsBuilder();
	auto actionParser = new ContinuousActionParser(TICK_SKIP);
	auto stateSetter = new RandomState(true, true, true);

	Match* match = new Match(
		rewards,
		terminalConditions,
		obs,
		actionParser,
		stateSetter,

		1, // Team size
		true // Spawn opponents
	);

	Gym* gym = new Gym(match, TICK_SKIP);
	return { match, gym };
}

int main() {
	// Initialize RocketSim with collision meshes
	RocketSim::Init("./collision_meshes");

	// Make configuration for the learner
	LearnerConfig cfg = {};

	// Play around with these to see what the optimal is for your machine, more isn't always better
	cfg.numThreads = 4;
	cfg.numGamesPerThread = 4;

	// We want a large itr/batch size
	// You'll want to increase this as your bot improves, up to an extent
	int tsPerItr = 4 * 1000;
	cfg.timestepsPerIteration = tsPerItr;
	cfg.ppo.batchSize = tsPerItr;
	cfg.ppo.miniBatchSize = 1 * 1000; // Must divide batchSize (4000 % 1000 == 0)
	cfg.expBufferSize = tsPerItr * 3;
	
	// This is just set to 1 to match rlgym-ppo example
	// I've found the best value is somewhere between 2 and 4
	// Increasing this will lower SPS, but increase step efficiency
	cfg.ppo.epochs = 1; 

	// Reasonable starting entropy
	cfg.ppo.entCoef = 0.01f;

	// Smoke run: stop after a single collect+learn iteration so we can verify the
	// Perceiver continuous learner loop runs end-to-end (collection + PPO update).
	cfg.timestepLimit = tsPerItr;

	// Good learning rate to start, may start to be too high around 100m steps
	cfg.ppo.policyLR = 2e-4;
	cfg.ppo.criticLR = 2e-4;

	// EARL Perceiver architecture. These MUST match what TemporalObsBuilder emits with
	// its default config (maxPlayers=4, nFrames=13, maxBoostPads=34):
	//   E (per-frame entity slots) = 1(ball) + maxPlayers*2(cars) + maxBoostPads = 1+8+34 = 43
	//   nTokens = F*E = 13*43 = 559
	//   queryFeatures = 39(self)+12(ball)+0(mode)+6(mutator)+2(team-size) = 59
	//   kvFeatures    = 30 (entity feature dims)
	//   obsSize       = 59 + 559*30 + 559 = 17388
	// (The BC checkpoint's main BotSpec uses a wider 208/69/151 layout; the parity gate
	//  exercises that layout against the real checkpoint. This example runs from random
	//  init at the builder's native width just to smoke-test the learner loop.)
	cfg.ppo.perceiver = PerceiverConfig{};
	cfg.ppo.perceiver.queryFeatures = 59;
	cfg.ppo.perceiver.kvFeatures = 30;
	cfg.ppo.perceiver.nTokens = 559;
	
	cfg.sendMetrics = false; // Smoke run: no metrics server needed
	cfg.renderMode = false; // Don't render

	// Make the learner with the environment creation function and the config we just made
	Learner learner = Learner(EnvCreateFunc, cfg);

	// Set up our callbacks
	learner.stepCallback = OnStep;
	learner.iterationCallback = OnIteration;

	// Start learning!
	learner.Learn();

	return 0;
}