#!/usr/bin/env Rscript
# Pooled topic shifts of the topic-by-group fits (I): weighted vs thinned (draw 1). Run from output/5_inference_final.
# Writes v3sub1_topic_shift_comparison.csv.
suppressMessages({library(brms);library(dplyr);library(bayestestR)})
sh <- function(f){ d <- as.data.frame(as_draws_df(readRDS(f))); cols <- grep("^r_topic\\[.*,synthetic\\]$", names(d), value=TRUE)
  bind_rows(lapply(cols, function(cn){ x<-d[[cn]]; h<-hdi(x,ci=.95); data.frame(topic=sub("^r_topic\\[(.*),synthetic\\]$","\\1",cn), or=exp(median(x)), lcl=exp(h$CI_low), ucl=exp(h$CI_high), pd=max(mean(x>0),mean(x<0)))})) }
out <- list()
for (m in c("ER","IP","EX")) { a <- sh(sprintf("models_bayesian/v3_%s_I_topic_by_group.rds",m)); b <- sh(sprintf("models_bayesian/sub1_v3_%s_I_topic_by_group.rds",m))
  out[[m]] <- inner_join(a,b,by="topic",suffix=c("_w","_s")) %>% mutate(mech=m, cred_w=pd_w>.975&(lcl_w>1|ucl_w<1), cred_s=pd_s>.975&(lcl_s>1|ucl_s<1)) }
r <- bind_rows(out)
write.csv(r, "v3sub1_topic_shift_comparison.csv", row.names = FALSE)
options(width=250, dplyr.print_max=100)
cat("credible weighted:", sum(r$cred_w), " sub:", sum(r$cred_s), " flips:", sum(r$cred_w!=r$cred_s), " max |dlogOR|:", round(max(abs(log(r$or_s/r$or_w))),3), "\n")
print(r %>% filter(cred_w|cred_s) %>% transmute(mech,topic,or_w=round(or_w,2),ci_w=sprintf("%.2f-%.2f",lcl_w,ucl_w),pd_w=round(pd_w,3),or_s=round(or_s,2),ci_s=sprintf("%.2f-%.2f",lcl_s,ucl_s),pd_s=round(pd_s,3)) %>% as.data.frame())
