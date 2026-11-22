#pragma once

#include "Threading/GameInst.h"
#include "Util/WelfordRunningStat.h"
#include "Util/MetricSender.h"
#include "Util/RenderSender.h"
#include "Util/Timer.h"
#include "LearnerConfig.h"

namespace RLGPC {

	typedef std::function<void(class Learner*, std::vector<Report>&)> IterationCallback;

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	// Multi-member: one PPOLearner + one ExperienceBuffer + one return-stat per "member" (owner id).
	// A single-bot run is the N=1 case (one member, every slot owned by member 0); behavior is identical.
	class RG_IMEXPORT Learner {
	public:
		LearnerConfig config;

		// Indexed by member/owner id. Member 0's PPOLearner replaces the historical single `ppo`.
		std::vector<class PPOLearner*> ppos;
		std::vector<class ExperienceBuffer*> buffers;
		std::vector<WelfordRunningStat> returnStats;

		class ThreadAgentManager* agentMgr;
		EnvCreateFn envCreateFn;
		MetricSender* metricSender;
		RenderSender* renderSender;

		struct SkillTracker* skillTracker;

		int obsSize;
		int actionAmount;

		std::string runID = {};

		uint64_t
			totalTimesteps = 0,
			totalEpochs = 0;

		// Cross-iteration timer for RunIteration() (carries the collection start time across calls)
		Timer epochTimer = {};
		int64_t tsSinceSave = 0;

		Learner(EnvCreateFn envCreateFunc, LearnerConfig config);
		void Learn();

		// Single collect+learn iteration. Learn() is just a loop over this. Returns one Report per
		// member so an external (Python) driver can interleave multiple members (Jacobi league: freeze
		// all -> collect -> update each). Behavior is identical to the inlined single-bot loop body.
		std::vector<Report> RunIteration();

		// Start/stop the background collection agents without running the learn loop (for drivers
		// that call RunIteration() themselves).
		void StartAgents();
		void StopAgents();

		void AddNewExperience(class GameTrajectory& gameTraj, int member, Report& report);

		void UpdateLearningRates(float policyLR, float criticLR);

		std::vector<Report> GetAllGameMetrics();

	void Save();
	void Load();
	void SaveStats(std::filesystem::path path, int member);
	void LoadStats(std::filesystem::path path, int member);

	// Cross-version BC warm-start: seed member `member`'s actor/critic/FSP from the WEIG blobs in
	// `folder` (PPO_POLICY.lt etc., produced by tools/convert_bc_checkpoint.py). Optimizers stay fresh.
	void SeedMember(int member, std::filesystem::path folder);

	// Runtime curriculum setters. These mutate the LIVE envs in place (each game's TeamConfigMutator
	// SpawnHolder / ComposedReward member spec) so the metric-gated driver can fade the reward scaffold or
	// recede the spawn distance mid-run WITHOUT rebuilding the learner. Games whose setter/reward isn't the
	// league variant (e.g. plain RandomState self-play) are skipped (no-op). Locks each agent's game-step
	// mutex so the mutation is atomic w.r.t. the collection worker.
	void SetSpawnCurriculum(bool enabled, float ratio, float prob);
	void SetMemberReward(int member, const std::vector<std::pair<std::string, float>>& termWeights,
	                     float teamSpirit, float opponentPunish);

		IterationCallback iterationCallback = NULL;
		StepCallback stepCallback = NULL;

		RG_NO_COPY(Learner);

		~Learner();
	};
}
