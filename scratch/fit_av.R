library(FangPsychometric)
library(dplyr)
library(forcats)
library(rethinking)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

av_dat <- audiovisual_binomial %>%
  filter(trial %in% c("pre", "post1"),
         rid != "av-post1-O-f-CE") %>%
  mutate_at(vars(rid, sid, trial, age_group), fct_drop) %>%
  mutate(trt = as.integer(trial != "pre"),
         x = soa / 1000) %>%
  rename(G = age_group, S = sid)

m1 <- ulam(alist(
  k ~ binomial_logit(n, p),
  p <- exp(b + bG[G] + (bT + bTG[G]) * trt) * (x - (a + aG[G] + (aT + aTG[G]) * trt)),

  a ~ normal(0, 0.1),
  aT ~ normal(0, 0.1),
  aG[G] ~ normal(0, sd_aG),
  aTG[G] ~ normal(0, sd_aTG),

  b ~ normal(4, 2),
  bT ~ normal(0, 2),
  bG[G] ~ normal(0, sd_bG),
  bTG[G] ~ normal(0, sd_bTG),

  sd_aG ~ half_cauchy(0, 1),
  sd_aTG ~ half_cauchy(0, 1),
  sd_bG ~ half_cauchy(0, 2.5),
  sd_bTG ~ half_cauchy(0, 2.5)
), data = av_dat, chains = 4, cores = 4, iter = 5000, log_lik = TRUE,
control = list(adapt_delta=0.95, max_treedepth=12))

m2 <- ulam(alist(
  k ~ binomial_logit(n, p),
  p <- exp(b + bG[G] + bS[S] + (bT + bTG[G] + bTS[S]) * trt) * (x - (a + aG[G] + aS[S] + (aT + aTG[G] + aTS[S]) * trt)),

  a ~ normal(0, 0.1),
  aT ~ normal(0, 0.1),
  aG[G] ~ normal(0, sd_aG),
  aS[S] ~ normal(0, sd_aS),
  aTG[G] ~ normal(0, sd_aTG),
  aTS[S] ~ normal(0, sd_aTS),

  b ~ normal(4, 2),
  bT ~ normal(0, 2),
  bG[G] ~ normal(0, sd_bG),
  bS[S] ~ normal(0, sd_bS),
  bTG[G] ~ normal(0, sd_bTG),
  bTS[S] ~ normal(0, sd_bTS),

  sd_aG ~ half_cauchy(0, 1),
  sd_aS ~ half_cauchy(0, 1),
  sd_aTG ~ half_cauchy(0, 1),
  sd_aTS ~ half_cauchy(0, 1),

  sd_bG ~ half_cauchy(0, 2.5),
  sd_bS ~ half_cauchy(0, 2.5),
  sd_bTG ~ half_cauchy(0, 2.5),
  sd_bTS ~ half_cauchy(0, 2.5)

), data = av_dat, chains = 4, cores = 4, iter = 5000, log_lik = TRUE,
control = list(adapt_delta=0.95, max_treedepth=12))

m3 <- ulam(alist(
  k ~ binomial_logit(n, p),
  p <- lG[G] + (1 - 2*lG[G]) * exp(b + bG[G] + (bT + bTG[G]) * trt) * (x - (a + aG[G] + (aT + aTG[G]) * trt)),

  lG[G] ~ beta(4, 96),

  a ~ normal(0, 0.1),
  aT ~ normal(0, 0.1),
  aG[G] ~ normal(0, sd_aG),
  aTG[G] ~ normal(0, sd_aTG),

  b ~ normal(4, 2),
  bT ~ normal(0, 2),
  bG[G] ~ normal(0, sd_bG),
  bTG[G] ~ normal(0, sd_bTG),

  sd_aG ~ half_cauchy(0, 1),
  sd_aTG ~ half_cauchy(0, 1),
  sd_bG ~ half_cauchy(0, 2.5),
  sd_bTG ~ half_cauchy(0, 2.5)
),
data = av_dat,
chains = 4, cores = 4, iter = 5000, log_lik = TRUE,
control = list(adapt_delta=0.95, max_treedepth=12),
start = list(lG = c(0.02, 0.02, 0.02)))

m4 <- ulam(alist(
  k ~ binomial_logit(n, p),
  p <- lG[G] + (1 - 2*lG[G]) * exp(b + bG[G] + bS[S] + (bT + bTG[G] + bTS[S]) * trt) * (x - (a + aG[G] + aS[S] + (aT + aTG[G] + aTS[S]) * trt)),

  lG[G] ~ beta(4, 96),

  a ~ normal(0, 0.1),
  aT ~ normal(0, 0.1),
  aG[G] ~ normal(0, sd_aG),
  aS[S] ~ normal(0, sd_aS),
  aTG[G] ~ normal(0, sd_aTG),
  aTS[S] ~ normal(0, sd_aTS),

  b ~ normal(4, 2),
  bT ~ normal(0, 2),
  bG[G] ~ normal(0, sd_bG),
  bS[S] ~ normal(0, sd_bS),
  bTG[G] ~ normal(0, sd_bTG),
  bTS[S] ~ normal(0, sd_bTS),

  sd_aG ~ half_cauchy(0, 1),
  sd_aS ~ half_cauchy(0, 1),
  sd_aTG ~ half_cauchy(0, 1),
  sd_aTS ~ half_cauchy(0, 1),

  sd_bG ~ half_cauchy(0, 2.5),
  sd_bS ~ half_cauchy(0, 2.5),
  sd_bTG ~ half_cauchy(0, 2.5),
  sd_bTS ~ half_cauchy(0, 2.5)

), data = av_dat, chains = 4, cores = 4, iter = 5000, log_lik = TRUE,
control = list(adapt_delta=0.95, max_treedepth=12),
start = list(lG = c(0.02, 0.02, 0.02)))

m5 <- ulam(alist(
  k ~ binomial_logit(n, p),
  p <- l + (1 - 2*l) * exp(b + bG[G] + bS[S] + (bT + bTG[G] + bTS[S]) * trt) * (x - (a + aG[G] + aS[S] + (aT + aTG[G] + aTS[S]) * trt)),

  l ~ beta(4, 96),

  a ~ normal(0, 0.1),
  aT ~ normal(0, 0.1),
  aG[G] ~ normal(0, sd_aG),
  aS[S] ~ normal(0, sd_aS),
  aTG[G] ~ normal(0, sd_aTG),
  aTS[S] ~ normal(0, sd_aTS),

  b ~ normal(4, 2),
  bT ~ normal(0, 2),
  bG[G] ~ normal(0, sd_bG),
  bS[S] ~ normal(0, sd_bS),
  bTG[G] ~ normal(0, sd_bTG),
  bTS[S] ~ normal(0, sd_bTS),

  sd_aG ~ half_cauchy(0, 1),
  sd_aS ~ half_cauchy(0, 1),
  sd_aTG ~ half_cauchy(0, 1),
  sd_aTS ~ half_cauchy(0, 1),

  sd_bG ~ half_cauchy(0, 2.5),
  sd_bS ~ half_cauchy(0, 2.5),
  sd_bTG ~ half_cauchy(0, 2.5),
  sd_bTS ~ half_cauchy(0, 2.5)

), data = av_dat, chains = 4, cores = 4, iter = 5000, log_lik = TRUE,
control = list(adapt_delta=0.95, max_treedepth=12),
start = list(l = 0.02))

m6 <- ulam(alist(
  k ~ binomial_logit(n, p),
  p <- l + (1 - 2*l) * exp(b + bG[G] + (bT + bTG[G]) * trt) * (x - (a + aG[G] + (aT + aTG[G]) * trt)),

  l ~ beta(4, 96),

  a ~ normal(0, 0.1),
  aT ~ normal(0, 0.1),
  aG[G] ~ normal(0, sd_aG),
  aTG[G] ~ normal(0, sd_aTG),

  b ~ normal(4, 2),
  bT ~ normal(0, 2),
  bG[G] ~ normal(0, sd_bG),
  bTG[G] ~ normal(0, sd_bTG),

  sd_aG ~ half_cauchy(0, 1),
  sd_aTG ~ half_cauchy(0, 1),

  sd_bG ~ half_cauchy(0, 2.5),
  sd_bTG ~ half_cauchy(0, 2.5)

), data = av_dat, chains = 4, cores = 4, iter = 5000, log_lik = TRUE,
control = list(adapt_delta=0.95, max_treedepth=12),
start = list(l = 0.02))

compare(m1, m2, m3, m4, m5, m6)

compare(m2, m4)
compare(m4, m5)
