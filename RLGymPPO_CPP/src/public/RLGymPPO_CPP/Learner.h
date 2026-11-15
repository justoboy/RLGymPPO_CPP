#pragma once

#include "Threading/GameInst.h"
#include "Util/WelfordRunningStat.h"
#include "Util/MetricSender.h"
#include "Util/RenderSender.h"
#include "Util/Timer.h"
#include "LearnerConfig.h"

namespace RLGPC {

	typedef std::function<void(class Learner*, Report&)> IterationCallback;

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	class RG_IMEXPORT Learner {
	public:
		LearnerConfig config;

		class PPOLearner* ppo;
		class ThreadAgentManager* agentMgr;
		class ExperienceBuffer* expBuffer;
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
			
		WelfordRunningStat returnStats = WelfordRunningStat(1);

		// Cross-iteration timer for RunIteration() (carries the collection start time across calls)
		Timer epochTimer = {};
		int64_t tsSinceSave = 0;

		Learner(EnvCreateFn envCreateFunc, LearnerConfig config);
		void Learn();

		// Single collect+learn iteration. Learn() is just a loop over this. Exposed so an external
		// (Python) driver can drive one bot's learner step-by-step and interleave multiple learners.
		Report RunIteration();

		// Start/stop the background collection agents without running the learn loop (for drivers
		// that call RunIteration() themselves).
		void StartAgents();
		void StopAgents();

		void AddNewExperience(class GameTrajectory& gameTraj, Report& report);

		void UpdateLearningRates(float policyLR, float criticLR);

		std::vector<Report> GetAllGameMetrics();

		void Save();
		void Load();
		void SaveStats(std::filesystem::path path);
		void LoadStats(std::filesystem::path path);

		IterationCallback iterationCallback = NULL;
		StepCallback stepCallback = NULL;

		RG_NO_COPY(Learner);

		~Learner();
	};
}