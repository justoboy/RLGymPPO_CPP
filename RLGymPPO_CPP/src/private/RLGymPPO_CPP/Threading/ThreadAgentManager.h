#pragma once
#include "ThreadAgent.h"
#include "../PPO/ExperienceBuffer.h"
#include <RLGymPPO_CPP/Util/Report.h>
#include <RLGymPPO_CPP/Util/WelfordRunningStat.h>
#include <RLGymPPO_CPP/Util/Timer.h>
#include <RLGymPPO_CPP/Util/RenderSender.h>

namespace RLGPC {
	// Multi-policy match primitive. The env always emits the SUPERSET (profile-on) flat obs; each policy
	// (one per "member"/owner) owns the slots stamped in the per-episode owner map (GameState::slotOwners,
	// read per game inside ThreadAgent). At each step we column-project the superset obs down to each
	// owner's own width (colIdx[g]) and run that owner's policy on just its slots, then scatter the
	// per-slot actions back. N=1 (one policy, identity colIdx) collapses to the old single-policy path.
	class ThreadAgentManager {
	public:
		// One policy per owner (member). policies[g] serves every slot whose owner id equals g.
		std::vector<GaussianPolicy*> policies;

		int nControls = 0; // shared action width across all owners

		// Per-owner column gather (int64 CPU) projecting the superset flat obs to owner g's width.
		// Undefined when the owner's width equals the superset (identity) -> skip the gather.
		std::vector<torch::Tensor> colIdx;

		std::vector<ThreadAgent*> agents;
		std::mutex expBufferMutex = {};
		std::mutex inferMutex = {};
		bool standardizeOBS;
		bool deterministic;
		bool blockConcurrentInfer;
		uint64_t maxCollect;
		torch::Device device;

		// Superset layout constants (derived from the widest policy config), used to build colIdx.
		int supEf = 0, supQ = 0, supKv = 0, supTokens = 0;

		// Single-policy mode: every slot belongs to owner 0 and the projection is identity. Lets the run
		// loop skip the per-owner gather/scatter and call policies[0]->GetAction(fullObs) directly.
		bool singlePolicy = false;

		RenderSender* renderSender = NULL;
		bool renderDuringTraining = false;
		float renderTimeScale = 1.f;

		bool disableCollection = false; // Prevents new steps from being started

		Timer iterationTimer = {};
		double lastIterationTime = 0;
		WelfordRunningStat obsStats;

		// Multi-policy constructor. policies is indexed by owner id. The per-episode slot->owner routing
		// lives in each GameInst (read from GameState::slotOwners), not here. Superset layout is taken
		// from the policy with the largest entityFeatures.
		ThreadAgentManager(
			std::vector<GaussianPolicy*> policies,
			bool standardizeOBS, bool deterministic, bool blockConcurrentInfer, uint64_t maxCollect, torch::Device device) :
			policies(policies),
			standardizeOBS(standardizeOBS), deterministic(deterministic), blockConcurrentInfer(blockConcurrentInfer),
			maxCollect(maxCollect), device(device) {

			// A single policy owns every slot => identity projection, byte-identical to the historical
			// single-policy path (no gather/scatter, GetAction called directly on the full obs tensor).
			singlePolicy = (policies.size() == 1);
			nControls = (int)policies[0]->config.nControls;

			// Superset = the widest owner layout. All owners share nTokens.
			supTokens = policies[0]->config.nTokens;
			supEf = 0;
			for (auto* p : policies)
				supEf = RS_MAX(supEf, (int)p->config.entityFeatures);
			for (auto* p : policies) {
				if ((int)p->config.entityFeatures == supEf) {
					supQ = (int)p->config.queryFeatures;
					supKv = (int)p->config.kvFeatures;
					break;
				}
			}

			for (auto* p : policies)
				this->colIdx.push_back(BuildColIdx((int)p->config.entityFeatures));
		}

		RG_NO_COPY(ThreadAgentManager);

		// Column gather from the superset flat obs to an owner with entity features `ef`. Returns an
		// undefined tensor when ef == supEf (identity: caller must skip the gather).
		// Superset flat obs = q(supQ) ++ kv(supTokens*supKv) ++ m(supTokens) ++ mode(1), where the
		// superset q = [q_self(supEf-5) ++ tail]. Owner's q = [q_self(ef-5) ++ tail]; owner's kv token
		// = [entity(ef) ++ rel(25)]; mask unchanged; mode = the single trailing column.
		torch::Tensor BuildColIdx(int ef) {
			if (ef == supEf)
				return torch::Tensor();

			const int supQSelf = supEf - 5; // superset q_self width (tail starts here)
			std::vector<int64_t> idx;
			// owner's q_self = superset columns [5, ef) (the non-IS_* self cols)
			for (int c = 5; c < ef; c++) idx.push_back(c);
			// shared tail (identical for every owner): superset columns [supQSelf, supQ)
			for (int c = supQSelf; c < supQ; c++) idx.push_back(c);
			for (int t = 0; t < supTokens; t++) {
				int64_t base = (int64_t)supQ + (int64_t)t * supKv;
				for (int c = 0; c < ef; c++) idx.push_back(base + c);       // owner's entity block (IS_* kept)
				for (int c = supEf; c < supKv; c++) idx.push_back(base + c); // relative block (25)
			}
			int64_t maskBase = (int64_t)supQ + (int64_t)supTokens * supKv;
			for (int t = 0; t < supTokens; t++) idx.push_back(maskBase + t);
			idx.push_back(maskBase + supTokens); // mode (trailing scalar channel)

			return torch::from_blob(idx.data(), { (int64_t)idx.size() }, torch::TensorOptions().dtype(torch::kLong)).clone();
		}

		void CreateAgents(EnvCreateFn func, int amount, int gamesPerAgent);

		void StartAgents() {
			for (ThreadAgent* agent : agents)
				agent->Start();
		}

		void StopAgents() {
			for (ThreadAgent* agent : agents)
				agent->Stop();
		}

		void SetStepCallback(StepCallback callback) {
			for (ThreadAgent* agent : agents)
				for (GameInst* game : agent->gameInsts)
					game->stepCallback = callback;
		}

		void GetMetrics(Report& report);
		void ResetMetrics();

		// Aggregate every game's per-owner outcome tallies (goals/touches) into one Report per owner.
		// Fills reports[g]["goals_for_mean"] / ["touches_for_mean"] with the mean per-episode count for
		// owner g across all games this iteration (NAN if owner g completed no episodes). The metric-gated
		// curriculum driver reads these, NOT reward magnitude.
		void GetOwnerMetrics(std::vector<Report>& reports);

		// Collect `amount` timesteps, returning one GameTrajectory per owner (indexed by owner id).
		std::vector<GameTrajectory> CollectTimesteps(uint64_t amount);

		~ThreadAgentManager() {
			for (ThreadAgent* agent : agents)
				delete agent;
		}
	};
}
