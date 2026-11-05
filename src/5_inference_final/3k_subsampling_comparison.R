#!/usr/bin/env Rscript
# Subsampling sensitivity check: compares the weighted strategy fits (v3_*) with draw 1 of the
# thinned-data fits (v3sub1_*, run: SUBS_SEED=1 MECHS=<m> STAGES=main Rscript 3j_strategy_models.R).
# Run from output/5_inference_final. Writes v3sub1_comparison_group_effects.csv.
suppressMessages(library(dplyr))
options(width=250, dplyr.width=Inf, dplyr.print_max=400, pillar.print_max=400)
rd <- function(p) bind_rows(lapply(c("ER","IP","EX"), function(m) read.csv(sprintf(p, m))))
w <- read.csv("v3_group_effects.csv")
s <- rd("v3sub1_%s_group_effects.csv")
key <- c("spec","mechanism","level","who","contrast")
m <- inner_join(w, s, by=key, suffix=c("_w","_s")) %>%
  mutate(cred_w = pd_w>0.975 & (lcl_w>1|ucl_w<1), cred_s = pd_s>0.975 & (lcl_s>1|ucl_s<1),
         lratio = log(or_s)-log(or_w), width_w=log(ucl_w/lcl_w), width_s=log(ucl_s/lcl_s))
write.csv(m, "v3sub1_comparison_group_effects.csv", row.names = FALSE)
cat("rows", nrow(w), nrow(s), nrow(m), "\n")
cat("\n== Summary by spec x level: n, flips in credibility, max |log OR diff|, median |diff|, median CI width ratio (sub/weighted)\n")
print(m %>% group_by(spec, level) %>% summarise(n=n(), flips=sum(cred_w!=cred_s), cred_w=sum(cred_w), cred_s=sum(cred_s),
  max_abs_dlogOR=round(max(abs(lratio)),3), med_abs=round(median(abs(lratio)),3), width_ratio=round(median(width_s/width_w),2), .groups="drop") %>% as.data.frame())
cat("\n== Flips\n")
print(m %>% filter(cred_w!=cred_s) %>% transmute(spec,mechanism,level,who,contrast,or_w=round(or_w,2),lcl_w=round(lcl_w,2),ucl_w=round(ucl_w,2),pd_w=round(pd_w,3),or_s=round(or_s,2),lcl_s=round(lcl_s,2),ucl_s=round(ucl_s,2),pd_s=round(pd_s,3)) %>% as.data.frame())
cat("\n== Opposite direction (sign) between weighted and sub, any\n")
print(m %>% filter(sign(log(or_w))!=sign(log(or_s))) %>% transmute(spec,mechanism,level,who,contrast,or_w=round(or_w,2),or_s=round(or_s,2),cred_w,cred_s) %>% as.data.frame())
cat("\n== Model-level gaps, length adjusted, side by side\n")
print(m %>% filter(spec=="length adjusted", level %in% c("model","strategy")) %>% transmute(mechanism,level,who,or_w=round(or_w,2),ci_w=sprintf("%.2f-%.2f",lcl_w,ucl_w),or_s=round(or_s,2),ci_s=sprintf("%.2f-%.2f",lcl_s,ucl_s)) %>% as.data.frame())
cat("\n== Severity by model, side by side\n")
print(m %>% filter(spec=="severity by group", level %in% c("model")) %>% transmute(mechanism,who,or_w=round(or_w,2),ci_w=sprintf("%.2f-%.2f",lcl_w,ucl_w),or_s=round(or_s,2),ci_s=sprintf("%.2f-%.2f",lcl_s,ucl_s)) %>% as.data.frame())
cat("\n== Strategy within model, length adjusted: credible in either\n")
print(m %>% filter(spec=="length adjusted", level=="strategy_within_model", cred_w|cred_s) %>% transmute(mechanism,who,contrast,or_w=round(or_w,2),ci_w=sprintf("%.2f-%.2f",lcl_w,ucl_w),or_s=round(or_s,2),ci_s=sprintf("%.2f-%.2f",lcl_s,ucl_s),cred_w,cred_s) %>% as.data.frame())
cat("\n== Length effect\n"); print(rbind(cbind(src="w",read.csv("v3_length_effect.csv")), cbind(src="sub",rd("v3sub1_%s_length_effect.csv"))) %>% mutate(across(where(is.numeric),~round(.x,2))))
cat("\n== SDs\n")
sw <- read.csv("v3_topic_by_group_sd.csv"); ss <- rd("v3sub1_%s_topic_by_group_sd.csv")
print(inner_join(sw, ss, by=c("mechanism","parameter"), suffix=c("_w","_s")) %>% transmute(mechanism,parameter,w=sprintf("%.3f (%.3f-%.3f)",median_w,lcl_w,ucl_w), sub=sprintf("%.3f (%.3f-%.3f)",median_s,lcl_s,ucl_s)) %>% as.data.frame())
cat("\n== Deviations: credible tm/tt/tmm\n")
dw <- read.csv("v3_topic_by_group_deviations.csv"); ds <- rd("v3sub1_%s_topic_by_group_deviations.csv")
cat("weighted credible:", sum(dw$credible), " sub credible:", sum(ds$credible), "\n")
print(bind_rows(cbind(src="w",dw%>%filter(credible)), cbind(src="sub",ds%>%filter(credible))) %>% transmute(src,mechanism,term,topic,model,strategy,or=round(or,2),lcl=round(lcl,2),ucl=round(ucl,2),pd=round(pd,3)) %>% as.data.frame())
