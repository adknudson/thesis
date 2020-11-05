library(FangPsychometric)
library(dplyr)
library(rstan)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
m <- stan_model(file = "models/universal_pf.stan")
n_chains <- 2L

keep_pars <- c(
  "a", "b", "lG",
  "aGT", "bGT",
  "aS", "bS",
  "sd_aGT", "sd_bGT",
  "sd_aS", "sd_bS",
  "log_lik", "k_pred"
)

obs_dat <- function(data) {
  dat <- data %>%
    filter(trial %in% c("pre", "post1")) %>%
    filter(rid != "av-post1-O-f-CE") %>%
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
  # k = k,
  # n = n,
  k = response,
  n = rep(1, length(response)),
  G = as.integer(age_group),
  trt = as.integer(trial),
  S = as.integer(sid)
))

init <- with(dat, replicate(n_chains, list(
  a_raw = rnorm(1),
  aGT_raw = matrix(rnorm(N_G * N_T, 0, 0.5), N_G, N_T),
  aS_raw = rnorm(N_S, 0, 0.5),
  aGT_unif = runif(1, 0, pi/4),
  aS_unif = runif(1, 0, pi/4),
  b_raw = rnorm(1),
  bGT_raw = matrix(rnorm(N_G * N_T, 0, 0.5), N_G, N_T),
  bS_raw = rnorm(N_S, 0, 0.5),
  bGT_unif = runif(1, 0, pi/4),
  bS_unif = runif(1, 0, pi/4),
  lG = runif(N_G, 0, 0.05)),
simplify = FALSE))

f <- sampling(
  object = m,
  data = stan_dat,
  chains = n_chains,
  cores = n_chains,
  iter = 15000,
  warmup = 5000,
  refresh = 500,
  init = init,
  thin = 10,
  control = list(adapt_delta = 0.95),
  pars = keep_pars
)

saveRDS(f, "models/m034s_sm.rds")
