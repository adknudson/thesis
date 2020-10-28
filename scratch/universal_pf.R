library(FangPsychometric)
library(dplyr)
library(rstan)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
m <- stan_model(file = "scratch/universal_pf.stan")

logit <- function(p) qlogis(p)
inv_logit <- function(x) plogis(x)
logistic <- function(x) inv_logit(x)
fn <- function(x, a, b) logistic(b * (x - a))

obs_dat <- function(data) {
  dat <- data %>%
    filter(trial %in% c("pre", "post1")) %>%
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

stan_summary <- function(object, ...) {
  round(summary(object, ...)$summary, 4)[,c(1,2,3,4,8,9,10)]
}

dat <- obs_dat(visual_binomial)
stan_dat <- with(dat, list(
  N = N,
  N_G = N_G,
  N_T = N_T,
  N_S = N_S,
  x = x,
  k = k,
  n = n,
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
  iter = 4000,
  warmup = 2000,
  refresh = 100,
  init = init,
  control = list(adapt_delta = 0.95),
  pars = keep_pars
)

stan_summary(f, c("a", "aG", "aT"))
stan_summary(f, c("aGT"))
stan_summary(f, c("b", "bG", "bT"))
stan_summary(f, c("bGT"))
stan_summary(f, "lG")
stan_summary(f, c("pss"))
stan_summary(f, c("jnd"))
stan_summary(f, pars = paste0("sd_a", c("G", "T", "GT", "S")))
stan_summary(f, pars = paste0("sd_b", c("G", "T", "GT", "S")))
stan_summary(f, "aS")
stan_summary(f, "bS")

post <- extract(f)
hist(post$lG[,3] - post$lG[,1], breaks = 50)
hist(post$lG[,3] - post$lG[,2], breaks = 50)
hist(post$lG[,1] - post$lG[,2], breaks = 50)

simulate_new_subject <- function(post, G, trt, method = c("average", "random")) {
  method <- match.arg(method)

  mu_alpha <- with(post, a + aG[,G] + aT[,trt] + aGT[,G,trt])
  mu_beta <- with(post, b + bG[,G] + bT[,trt] + bGT[,G,trt])

  if (method == "average") {
    alpha <- mu_alpha
    beta <- mu_beta
  } else {
    alpha <- rnorm(length(mu_alpha), mu_alpha, post$sd_aS)
    beta <- rnorm(length(mu_beta), mu_beta, post$sd_bS)
  }

  lambda <- with(post, lG[,G])

  list(alpha = alpha, beta = beta, lambda = lambda)
}

pre <- simulate_new_subject(post, 1, 1)
pos <- simulate_new_subject(post, 1, 2)
round(summary(pre$alpha), 4)
round(summary(pos$alpha), 4)
