#!/usr/bin/env bash
# Experts-only re-run of the paper pipeline (without laypeople / unverified users), see README.md.
# Run from src/5_inference_final:   nohup caffeinate -i bash run_expert_pipeline.sh > run_expert_pipeline.log 2>&1 &
# Every step logs to ../../output/5_inference_final_expert/logs/. Finished model fits are cached (.rds),
# so after an interruption the script can simply be started again: it resumes where it stopped.
set -euo pipefail
export EXCLUDE_LAYPEOPLE=1
OUT=../../output/5_inference_final_expert
LOG=$OUT/logs
mkdir -p "$LOG"
step() { echo "[$(date '+%F %T')] START $1"; }
done_() { echo "[$(date '+%F %T')] DONE  $1"; }

step "3i topic/severity models";            Rscript 3i_topic_llm_moderation_v2.R > "$LOG/3i.log" 2>&1;            done_ 3i
step "3j strategy models (weighted)";       Rscript 3j_strategy_models.R > "$LOG/3j.log" 2>&1;                    done_ 3j
for m in ER IP EX; do
  step "3j thinned fits $m";                SUBS_SEED=1 MECHS=$m Rscript 3j_strategy_models.R > "$LOG/3j_sub1_$m.log" 2>&1; done_ "3j thinned $m"
done
step "3k comparisons"
( cd "$OUT" && Rscript ../../src/5_inference_final/3k_subsampling_comparison.R > logs/3k_comparison.log 2>&1 \
            && Rscript ../../src/5_inference_final/3k_subsampling_topic_shifts.R > logs/3k_topic_shifts.log 2>&1 )
done_ 3k
step "descriptives";                        Rscript paper_descriptives.R > "$LOG/descriptives.log" 2>&1;           done_ descriptives
step "joint tests";                         Rscript paper_joint_tests.R > "$LOG/joint_tests.log" 2>&1;             done_ "joint tests"
step "paper";                               Rscript render_expert.R > "$LOG/paper.log" 2>&1;                         done_ paper
echo "[$(date '+%F %T')] PIPELINE FINISHED"
