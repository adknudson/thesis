library(FangPsychometric)
library(dplyr)
library(rstan)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
m <- stan_model(file = "models/universal_pf.stan")

obs_dat <- function(data) {
  dat <- data %>%
    filter(trial %in% c("pre", "post1")) %>%
    # filter(rid != "av-post1-O-f-CE") %>%
    mutate(x = soa / 1000,
           rid = factor(rid),
           sid = factor(sid),
           trial = factor(trial)) %>%
    as.list()
  dat$N <- length(dat$x)
  dat$N_G <- length(levels(dat$age_group))
  dat$N_S <- length(levels(dat$sid))
  dat$N_T <- length(levels(dat$trial))
  dat
}

dat <- obs_dat(sensorimotor)
stan_dat <- with(dat, list(
  N = N,
  N_G = N_G,
  N_T = N_T,
  N_S = N_S,
  x = x,
  k = response,
  n = rep(1, length(response)),
  G = as.integer(age_group),
  trt = as.integer(trial),
  S = as.integer(sid)
))


keep_pars <- c(
  "a", "b", "lG",
  "aG", "bG",
  "aT", "bT",
  "aGT", "bGT",
  "aS", "bS",
  "sd_aG", "sd_bG",
  "sd_aT", "sd_bT",
  "sd_aGT", "sd_bGT",
  "sd_aS", "sd_bS"
)

n_chains <- 2L

init <- replicate(n_chains, list(
  a_raw = rnorm(1),
  aG_raw = rnorm(dat$N_G, 0, 0.5),
  aT_raw = rnorm(dat$N_T, 0, 0.5),
  aGT_raw = matrix(rnorm(dat$N_G * dat$N_T, 0, 0.5), dat$N_G, dat$N_T),
  aS_raw = rnorm(dat$N_S, 0, 0.5),
  aG_unif = runif(1, 0, pi/4),
  aT_unif = runif(1, 0, pi/4),
  aGT_unif = runif(1, 0, pi/4),
  aS_unif = runif(1, 0, pi/4),
  b_raw = rnorm(1),
  bG_raw = rnorm(dat$N_G, 0, 0.5),
  bT_raw = rnorm(dat$N_T, 0, 0.5),
  bGT_raw = matrix(rnorm(dat$N_G * dat$N_T, 0, 0.5), dat$N_G, dat$N_T),
  bS_raw = rnorm(dat$N_S, 0, 0.5),
  bG_unif = runif(1, 0, pi/4),
  bT_unif = runif(1, 0, pi/4),
  bGT_unif = runif(1, 0, pi/4),
  bS_unif = runif(1, 0, pi/4),
  lG = runif(dat$N_G, 0, 0.05)),
simplify = FALSE)

f <- sampling(
  object = m,
  data = stan_dat,
  chains = n_chains,
  cores = n_chains,
  iter = 12000,
  warmup = 2000,
  refresh = 200,
  init = init,
  control = list(adapt_delta = 0.95),
  pars = keep_pars
)

saveRDS(f, "models/m034s_sm.rds")
